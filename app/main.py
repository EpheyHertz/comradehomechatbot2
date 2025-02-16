from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .schemas import ChatRequest
from .config import SECRET_KEY
from chatbot.chat import real_estate_chatbot

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

# Favicon Route
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
     return FileResponse("static/favicon.ico")

# Root Endpoint
@app.get("/", tags=["Welcome"])
def read_root():
    return {"message": "Welcome to the House Listing Chatbot API"}

# Chat Endpoint
@app.post("/chat/us", status_code=status.HTTP_200_OK, tags=["Chat With Us"])
async def chat_endpoint(request: ChatRequest):
    try:
        response = real_estate_chatbot(request.message)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred! Please try again later!"
        )
    return {"response": response}

# Ensure this script runs properly when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
