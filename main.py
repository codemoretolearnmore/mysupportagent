from fastapi import FastAPI, UploadFile, File, Request, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

origins = [
    "http://localhost:3000",  # Allow frontend during development
    "http://127.0.0.1:3000",  # If accessing via 127.0.0.1
    "https://yourfrontend.com",  # Add your production frontend domain here
]

app = FastAPI(
    title="Support Ticket Classification API",
    description="API for classifying and managing support tickets with AI.",
    version="1.0.0"
)


@app.get("/")
async def root(request:Request):
    
    return {"message": "Server is running!"}



if __name__ == "__main__":
    print("🚀 Server is starting on http://0.0.0.0:8000 ...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
