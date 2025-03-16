#a fastapi schema for the chat request
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message")
    thread_id: Optional[str] = Field(None, description="The conversation thread ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Show me apartments in New York under $3000",
                "thread_id": ""
            }
        }

