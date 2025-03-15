import logging
import uuid
from fastapi import Request

# Define the logging format
LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(request_id)s] - %(message)s"

# Custom filter to ensure `request_id` is always present
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "N/A"  # Default value if missing
        return True

# Create a logger
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)  # Set to INFO in production

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler("app.log", mode="a")
file_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for both handlers
formatter = logging.Formatter(LOG_FORMAT)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Attach the custom request_id filter
logger.addFilter(RequestIdFilter())

# Middleware to attach `request_id` to every request
async def request_id_middleware(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())  # Generate request ID
    response = await call_next(request)
    return response

# Function to get logger with request_id
def get_logger(request: Request):
    request_id = getattr(request.state, "request_id", "N/A")
    return logging.LoggerAdapter(logger, {"request_id": request_id})
