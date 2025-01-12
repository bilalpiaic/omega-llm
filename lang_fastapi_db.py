from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import slowapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Database Setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./ai_chat.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message = Column(Text)
    response = Column(Text)
    context = Column(Text)

Base.metadata.create_all(bind=engine)

# Pydantic Models
class ChatMessage(BaseModel):
    message: str
    context: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    timestamp: datetime

# API Key Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")
API_KEYS = {os.getenv("API_KEY"): "default-user"}

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(slowapi.errors.RateLimitExceeded, _rate_limit_exceeded_handler)

# Complex Prompt Templates
SYSTEM_TEMPLATE = """
You are an AI assistant with the following capabilities and constraints:

Context: {context}
Previous Conversation: {history}
Current Time (UTC): {current_time}

Guidelines:
1. Provide detailed, accurate responses
2. Maintain conversation context
3. Be concise yet informative
4. Cite sources when applicable
5. Address the user's specific needs

User Query: {message}

Response:
"""

# LangGraph Workflow States
class WorkflowState(BaseModel):
    input: str
    context: Optional[str]
    processed_input: Optional[str]
    enhanced_context: Optional[str]
    response: Optional[str]
    memory: Optional[dict]

# Enhanced LangChain Setup
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
        verbose=True
    )

# Database Dependencies
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication
def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return API_KEYS[api_key]

# Conversation Memory
class EnhancedConversationMemory(ConversationBufferMemory):
    def __init__(self):
        super().__init__()
        self.conversations = []

    def add_message(self, role: str, content: str):
        self.conversations.append({"role": role, "content": content})

    def get_history(self) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" 
                         for msg in self.conversations[-5:]])  # Last 5 messages

# LangGraph Workflow
def create_workflow():
    workflow = StateGraph(WorkflowState)
    
    # Input Processing Node
    def process_input(state: WorkflowState) -> Dict:
        state.processed_input = state.input.strip().lower()
        return state.dict()

    # Context Enhancement Node
    def enhance_context(state: WorkflowState) -> Dict:
        context = state.context or ""
        current_time = datetime.utcnow().isoformat()
        state.enhanced_context = f"{context}\nCurrent time: {current_time}"
        return state.dict()

    # Response Generation Node
    def generate_response(state: WorkflowState) -> Dict:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
        
        history = state.memory.get("history", "") if state.memory else ""
        
        response = llm(prompt.format(
            context=state.enhanced_context,
            history=history,
            current_time=datetime.utcnow().isoformat(),
            message=state.processed_input
        ))
        
        state.response = response.content
        return state.dict()

    # Add nodes and edges
    workflow.add_node("process_input", process_input)
    workflow.add_node("enhance_context", enhance_context)
    workflow.add_node("generate_response", generate_response)

    workflow.add_edge("process_input", "enhance_context")
    workflow.add_edge("enhance_context", "generate_response")
    workflow.add_edge("generate_response", END)

    workflow.set_entry_point("process_input")
    
    return workflow.compile()

# API Endpoints
@app.post("/chat", response_model=ChatResponse)
@limiter.limit("5/minute")
async def chat(
    message: ChatMessage,
    user_id: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    try:
        # Initialize workflow
        workflow = create_workflow()
        memory = EnhancedConversationMemory()

        # Get conversation history
        history = db.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(Conversation.timestamp.desc()).limit(5).all()

        # Prepare memory
        for conv in reversed(history):
            memory.add_message("user", conv.message)
            memory.add_message("assistant", conv.response)

        # Execute workflow
        result = workflow.invoke({
            "input": message.message,
            "context": message.context,
            "memory": {"history": memory.get_history()}
        })

        # Save to database
        conversation = Conversation(
            user_id=user_id,
            message=message.message,
            response=result["response"],
            context=message.context
        )
        db.add(conversation)
        db.commit()

        return ChatResponse(
            response=result["response"],
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/conversations/{user_id}")
async def get_conversations(
    user_id: str,
    limit: int = 10,
    _: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    conversations = db.query(Conversation).filter(
        Conversation.user_id == user_id
    ).order_by(Conversation.timestamp.desc()).limit(limit).all()
    
    return [{"message": conv.message, 
             "response": conv.response, 
             "timestamp": conv.timestamp} 
            for conv in conversations]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)