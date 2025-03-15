from fastapi import FastAPI, UploadFile, File, Request, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from services.ticket_classification import classify_tickets
from services.train_with_chatgpt import train_with_chatgpt
from services.model_training import train_model
import utils.connection as connection
import asyncio
from services.database import createClassificationJob, checkClassificationTaskStatus, getClassificationResults, updateCategoryClassification
import json
from typing import List
from pydantic import BaseModel
from utils.validateFile import isValidJSONFile, isAllColumnsPresent, isEmptyFile
from logging_config import request_id_middleware, get_logger
import logging
import uuid

origins = [
    "http://localhost:3000",  # Allow frontend during development
    "http://127.0.0.1:3000",  # If accessing via 127.0.0.1
    "https://mysupportagent-frontend-production.up.railway.app",  # Add your production frontend domain here
]

app = FastAPI(
    title="Support Ticket Classification API",
    description="API for classifying and managing support tickets with AI.",
    version="1.0.0"
)
app.middleware("http")(request_id_middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow only specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
async def root(request:Request):
    logger = get_logger(request)
    logger.info("Request to Index Server Received")
    return {"message": "Server is running!"}

class Ticket(BaseModel):
    ticket_id: int
    description: str
    created_date: str
    product: str

class TicketRequest(BaseModel):
    tickets: List[Ticket]

@app.post("/classify_tickets")
async def classify(request:Request, background_tasks: BackgroundTasks, file:UploadFile = File(...)):
    logger = get_logger(request)
    if isValidJSONFile(file)==False:
        logger.error("Invalid JSON File Format Received")
        return JSONResponse(
            status_code=400, content={"message":"Invalid file format. Please upload a JSON file.", job_id:""}
        )
    try:
        # Read file content
        file_content = await file.read()
        
        # Parse JSON data
        data = json.loads(file_content)
        
        # Extract tickets from JSON
        tickets = data.get("tickets", [])
        

        if isEmptyFile(tickets)==True:
            logger.warning("Empty JSON file received for classification")
            return JSONResponse(
            status_code=400, content={"message":"Uploaded file is empty", job_id:""}
        )
        if isAllColumnsPresent(tickets)==False:
            logger.warning("Missing columns in JSON file required for classification")
            return JSONResponse(
                status_code=400, content={"message":"uploaded file missing description and product column", job_id:""}
            )
        job_id = await createClassificationJob(logger)
        if job_id:
            background_tasks.add_task(classify_tickets, job_id, tickets, logger)
            return JSONResponse(
                status_code=200,
                content={"message":"Ticket Classification Started", "job_id":job_id}
            )
        else:
            raise HTTPException(status_code=500, detail="Internal Server")

        
    except Exception as e:
        logger.error("Exception occured while processing request")
        return JSONResponse(
            status_code=400, content={"message":str(e), "job_id":""}
        )
    
@app.websocket("/ws/classification/{job_id}")
async def websocket_classification(websocket: WebSocket, job_id: str):
    await websocket.accept()
    # logger = None  # Add your logging logic here if needed
    request_id = str(uuid.uuid4())  
    logger = logging.LoggerAdapter(logging.getLogger("app_logger"), {"request_id": request_id})
    logger.info(f"WebSocket connection established for job_id: {job_id}")
    try:
        while True:
            # Periodically check classification status
            response = await checkClassificationTaskStatus(logger, job_id)

            if response and response["status"]=='COMPLETED':
                results = await getClassificationResults(job_id, logger)

                # Send classified tickets once the job is complete
                await websocket.send_json({"message": "Classification Completed", "classified_tickets": results})
                break  # Exit loop after sending data
            elif response and response["status"]=='FAILED':
                await websocket.send_json({"message": "Classification FAILED", "classified_tickets": []})
                break  # Exit loop after sending data
            elif response is None:
                await websocket.send_json({"message": "No Classification Job Found", "classified_tickets": []})
                break
            await asyncio.sleep(5)  # Check status every 5 seconds

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for job_id: {job_id}")

    except Exception as e:
        await websocket.send_json({"error": "Internal Error"})
        print("Error:", e)


@app.get("/classified_tickets/{job_id}")
async def get_classified_tickets(job_id:str, request:Request):
    logger = get_logger(request)
    logger.info("Request for classification result received")
    print(job_id)
    results =  list(await getClassificationResults(job_id, logger))
    print(results)
    if results:
        return JSONResponse(
            status_code=200, content={"message":"Classified Tickets Fetched", "classified_tickets":results}
        )
    else:
        return JSONResponse(status_code=404, content={"message":"No Classified ticket found with this job", "classified_ticekts":[]})
    

@app.put("/save_edit")
async def save_edit(request:Request, data: dict):
    logger = get_logger(request)
    logger.info("Save request for edited ticket received")
    result = await updateCategoryClassification(data, logger)
    if result:
        return JSONResponse(status_code=200, content={"message":"Saved Ticket Classification","updatedTicket":[]})
    else:
        return JSONResponse(status_code=409, content={"message":"Couldn't update category", "updatedTicket":[]})



@app.post("/train_with_chatgpt")
async def train_chatgpt(request:Request, data: TicketRequest):
    logger = get_logger(request)
    logger.info("Request to label tickets with LLM model received")
    tickets = data.tickets
    if not tickets:
        logger.warning("Received empty json object of tickets")
        return JSONResponse(
            status_code=400,
            content={"message":"No tickets found for classification", "classified_tickets":[]}
        )
    print(tickets)
    return await train_with_chatgpt(tickets, logger)

@app.post("/train_model")
async def train(request:Request):
    logger = get_logger(request)
    logger.info("Request to train model received")
    return train_model(logger)
