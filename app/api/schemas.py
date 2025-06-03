from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatMessageInput(BaseModel):
    user_input: str
    session_id: Optional[str] = None # For future session management
    chat_history: List[Dict[str, str]] = Field(default_factory=list, description="List of {'role': 'human'/'ai', 'content': 'message'}")

class AgentResponse(BaseModel):
    output: str
    chat_history: List[Dict[str, str]]
    current_stage: Optional[str] = None
    contract_state_summary: Optional[Dict[str, Any]] = None # To give client insight

class AddTemplateForm(BaseModel): # For form data, file handled separately
    contract_type: str
    description: Optional[str] = ""

class RagSearchQuery(BaseModel):
    query: str
    contract_type: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)

class StatusResponse(BaseModel):
    status: Dict[str, Any]

class TemplateListResponse(BaseModel):
    templates_summary: str # Or a more structured list of objects

class ErrorResponse(BaseModel):
    error: str
    message: Optional[str] = None
    details: Optional[Any] = None