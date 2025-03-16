import logging
from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .schemas import ChatRequest
from .config import SECRET_KEY
from chatbot.chat import real_estate_chatbot,continue_conversation,inspect_state

app = FastAPI()

# Add Middleware
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Serve Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("real_estate_chatbot")
# Favicon Route
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
     return FileResponse("static/favicon.ico")

# Root Endpoint
@app.get("/", tags=["Welcome"])
def read_root():
    return {"message": "Welcome to the House Listing Chatbot API"}

# Chat Endpoint
# Updated API endpoint
@app.post("/chat/us", status_code=status.HTTP_200_OK, tags=["Chat With Us"])
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Received chat request with thread_id: '{request.thread_id}'")
        
        # Check if thread_id exists and is not empty after stripping whitespace
        if request.thread_id and request.thread_id.strip():
            # Use the trimmed thread_id for continuing conversation
            trimmed_thread_id = request.thread_id.strip()
            logger.info(f"Using trimmed thread_id: '{trimmed_thread_id}'")
            
            # Log the state before processing
            inspect_state(trimmed_thread_id)
            
            # Continue existing conversation
            response = continue_conversation(request.message, trimmed_thread_id)
        else:
            # Start a new conversation with a new thread_id
            logger.info("No valid thread_id provided, starting new conversation")
            response = real_estate_chatbot(request.message, None)
        
        # Log the final response
        logger.info(f"Sending response with thread_id: {response['thread_id']}")
        logger.info(f"Response message: {response['response'][:100]}...")  # Log first 100 chars
        logger.info(f"Message history has {len(response.get('message_history', []))} messages")
        
        return response
        
    except Exception as e:
        error_message = f"Chat endpoint error: {str(e)}"
        logger.error(error_message)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred! Please try again later!"
        )

# Add an endpoint to inspect the current state of a thread
@app.get("/chat/state/{thread_id}", status_code=status.HTTP_200_OK, tags=["Debug"])
async def get_chat_state(thread_id: str):
    try:
        logger.info(f"State inspection requested for thread_id: {thread_id}")
        message_history = inspect_state(thread_id)
        
        return {
            "thread_id": thread_id,
            "message_count": len(message_history),
            "messages": message_history
        }
    except Exception as e:
        error_message = f"State inspection error: {str(e)}"
        logger.error(error_message)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while inspecting the state!"
        )
# Ensure this script runs properly when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
