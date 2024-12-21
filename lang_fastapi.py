from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
import os

# Load environment variables
load_dotenv()

# Configure Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini model with LangChain
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        verbose=True
    )

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    Please answer the following question in a clear and concise manner:
    
    Question: {question}
    
    Answer:
    """
)

# Create LangChain chain
def get_chain():
    llm = get_llm()
    return LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True
    )

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to the AI Assistant API"}

@app.post("/ask")
async def ask_question(question: str):
    try:
        # Initialize the chain
        chain = get_chain()
        
        # Get response from the model
        response = chain.run(question=question)
        
        return {
            "question": question,
            "answer": response
        }
    except Exception as e:
        return {
            "error": str(e)
        }

# Advanced example using LangGraph (experimental)
from typing import Dict, TypeVar, List
from langgraph.graph import StateGraph, END

# Define state type
State = TypeVar("State", bound=Dict)

# Create a simple graph workflow
def create_workflow():
    # Create a new graph
    workflow = StateGraph(State)
    
    # Add nodes to the graph
    workflow.add_node("process_input", lambda state: {
        "processed_input": state["input"].strip().lower()
    })
    
    workflow.add_node("generate_response", lambda state: {
        "response": get_chain().run(question=state["processed_input"])
    })
    
    # Add edges
    workflow.add_edge("process_input", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Set entry point
    workflow.set_entry_point("process_input")
    
    return workflow.compile()

@app.post("/chat")
async def chat_workflow(message: str):
    try:
        # Create and run workflow
        workflow = create_workflow()
        result = workflow.invoke({
            "input": message
        })
        
        return {
            "input": message,
            "response": result["response"]
        }
    except Exception as e:
        return {
            "error": str(e)
        }

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# poetry run python uvicorn lang_fastapi:app --reload