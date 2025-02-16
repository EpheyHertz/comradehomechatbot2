#a fastapi schema for the chat request
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
class ChatRequest(BaseModel):
    message: str


