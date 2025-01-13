# Required library imports
from typing_extensions import TypedDict  # For creating typed dictionary
from typing import Annotated  # For annotating types
from langgraph.graph import StateGraph, START, END  # Core elements for graph structure
from langgraph.graph.message import add_messages  # Message annotation utility
from langgraph.checkpoint.mongodb import MongoDBSaver  # MongoDB checkpoint integration
from pymongo import MongoClient  # MongoDB client
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Gemini AI integration
from fastapi import FastAPI  # FastAPI for creating the web API

from dotenv import load_dotenv  # To load environment variables from .env file
import os  # For interacting with environment variables

# Load environment variables from .env file
load_dotenv()

# Initialize Google Gemini AI model with the API key from environment variables
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Step 1: Define a state for storing messages
# Using TypedDict to specify the expected structure of the state
class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]  # Annotated list of messages to track conversation history

# Step 2: Define the assistant function
# This function handles interaction with the AI model and returns a response
def assistant(state: MessagesState):
    # Invokes the AI model with the user's messages and returns the response
    return {"messages": [llm.invoke(state["messages"])]}

# Step 3: Build the state graph
# This creates the structure of the chatbot's conversational flow
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)  # Add the assistant node (function) to the graph
builder.add_edge(START, "assistant")  # Define the starting point of the graph
builder.add_edge("assistant", END)  # Define the end point of the graph

# Step 4: Integrate MongoDB for saving conversation states
mongodb_client = MongoClient(os.getenv("MONGODB_URI"))  # Initialize MongoDB client with URI from environment variables
memory = MongoDBSaver(mongodb_client)  # Create a saver to persist graph state into MongoDB

# Compile the graph with MongoDB-based checkpointing
graph = builder.compile(checkpointer=memory)

# Step 5: Create the FastAPI app
app = FastAPI()

# Step 6: Define the API endpoint for chat
@app.get("/chat/{query}")  # Route to handle GET requests at "/chat/{query}"
def get_content(query: str):
    print(query)  # Log the query for debugging purposes
    try:
        # Configuration for the graph, including thread ID for saving context
        config = {"configurable": {"thread_id": "1"}}
        
        # Invoke the graph with user query and configuration
        result = graph.invoke({"messages": [("user", query)]}, config)
        
        # Return the result as the response
        return result
    
    except Exception as e:  # Handle errors gracefully
        return {"output": str(e)}

# Step 7: Command to run the FastAPI app
# Use the following command in your terminal to run the app with Uvicorn
# poetry run - uvicorn short_term_memory_mongodb_persist:app --reload
