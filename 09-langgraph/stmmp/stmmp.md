# Design, Structure, and Algorithm (DSA) for Conversational Chatbot with MongoDB Persistence

## **Design**
The system is designed to implement a conversational chatbot API using:

1. **AI Integration**: Powered by Google Gemini AI to handle user queries and generate responses.
2. **State Management**: Tracks the conversation's history using a typed dictionary (`MessagesState`).
3. **Persistence Layer**: MongoDB stores conversation states for continuity across interactions.
4. **API Layer**: FastAPI provides a RESTful endpoint for chatbot interaction.

---

## **Structure**
The project is modular, with logical components as follows:

### **1. Library Imports**
- **Core Libraries**:
  - `TypedDict`, `Annotated` for defining typed data structures.
  - LangGraph for building conversation states and transitions.
  - MongoDB for persisting conversation data.
  - FastAPI for API creation.
- **AI Integration**:
  - `ChatGoogleGenerativeAI` to connect to Google Gemini AI for response generation.
- **Environment Configuration**:
  - `dotenv` for securely loading API keys and MongoDB credentials.

### **2. Environment Configuration**
- Load environment variables (e.g., `GOOGLE_API_KEY`, `MONGODB_URI`) using `.env` file.

### **3. AI and Conversation State**
- Define `MessagesState`:
  - A typed dictionary to store conversation history.
- Define the `assistant` function:
  - Invokes Google Gemini AI to generate a response based on user input.

### **4. State Graph**
- Build a `StateGraph`:
  - **Nodes**: Functional units (e.g., `assistant`).
  - **Edges**: Define transitions between nodes (e.g., `START -> assistant -> END`).

### **5. MongoDB Integration**
- Use `MongoDBSaver` to persist conversation states in MongoDB.
- Connect via `MongoClient` using the URI from environment variables.

### **6. API Creation**
- Use FastAPI to create the `/chat/{query}` endpoint.
- This endpoint invokes the graph, processes the user query, and returns the response.

### **7. Execution**
- Run the application using Uvicorn to serve HTTP requests.

---

## **Algorithm**

### **1. Initialization**
- Load environment variables.
- Initialize Google Gemini AI (`ChatGoogleGenerativeAI`).

### **2. Define Data Structure**
- Use `MessagesState` to store:
  - A list of user and assistant messages.

### **3. Assistant Function**
- Extract the user message from the state.
- Pass it to Google Gemini AI.
- Return the assistant's response in the updated state.

### **4. State Graph Compilation**
- Define nodes and transitions:
  - `START -> assistant -> END`.
- Integrate MongoDB to persist graph states using `MongoDBSaver`.

### **5. API Endpoint Logic**
- Accept user queries via the `/chat/{query}` endpoint.
- Configure the graph with a unique thread ID for each conversation.
- Pass user input to the graph and retrieve the AI-generated response.
- Return the result to the user.

### **6. Error Handling**
- Catch exceptions during graph execution or MongoDB interactions.
- Return meaningful error messages.

### **7. Deployment**
- Run the FastAPI app using Uvicorn for local or production use.

---

## **Execution Command**
To start the application, run the following command:

```bash
poetry run uvicorn short_term_memory_mongodb_persist:app --reload
```

This will start the FastAPI app and expose the `/chat/{query}` endpoint for handling user queries.

---

## **Key Features**
- **AI-Driven**: Uses Google Gemini AI for natural language understanding and response generation.
- **State Persistence**: MongoDB ensures conversation continuity across sessions.
- **Scalable API**: FastAPI provides a robust and performant interface for user interaction.
- **Modular Design**: Each component is independent, making the system easy to extend or modify.



# Code Workflow Documentation

This document provides a step-by-step explanation of the code's workflow and functionality.

## Workflow Overview

1. **Initialize Dependencies:**
   - Import the required libraries for building a chatbot using LangGraph, MongoDB, FastAPI, and Google Gemini AI.
   - Load environment variables from a `.env` file for secure configuration management.

2. **Setup AI Model:**
   - Use `ChatGoogleGenerativeAI` to initialize the Gemini AI model.
   - Retrieve the API key from the `.env` file.

3. **Define Conversation State:**
   - Create a `MessagesState` class using `TypedDict` to define the structure for storing user messages.

4. **Build the State Graph:**
   - Use `StateGraph` to define the chatbot's conversational flow.
   - Add nodes and edges to represent different states and transitions within the conversation.

5. **Integrate MongoDB:**
   - Initialize a MongoDB client using the URI from the `.env` file.
   - Set up `MongoDBSaver` to persist conversation states.
   - Compile the graph with MongoDB checkpointing enabled.

6. **Set Up API:**
   - Create a FastAPI application.
   - Define an endpoint (`/chat/{query}`) to process user queries and generate responses.

7. **Set Up Streamlit Frontend:**
   - Create a Streamlit application to serve as the frontend interface for the chatbot.

8. **Run the Application:**
   - Use the Uvicorn command to start the FastAPI app locally.
   - Run the Streamlit app to provide a user-friendly frontend.

## Step-by-Step Code Explanation

### Step 1: Initialize Dependencies

```python
# Import required libraries
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
```

### Step 2: Setup AI Model

```python
# Initialize Google Gemini AI model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
```

### Step 3: Define Conversation State

```python
# Define a state to store messages
class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]
```

### Step 4: Build the State Graph

```python
# Define the assistant function
def assistant(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

# Build the state graph
builder = StateGraph(MessagesState)

# Add nodes and edges
builder.add_node("assistant", assistant)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

# Integrate MongoDB for state persistence
mongodb_client = MongoClient(os.getenv("MONGODB_URI"))
memory = MongoDBSaver(mongodb_client)

# Compile the graph with MongoDB checkpointing
graph = builder.compile(checkpointer=memory)
```

### Step 5: Set Up API

```python
# Create the FastAPI app
app = FastAPI()

# Define the chat endpoint
@app.get("/chat/{query}")
def get_content(query: str):
    print(query)  # Log the query
    try:
        # Configuration for graph invocation
        config = {"configurable": {"thread_id": "1"}}
        
        # Invoke the graph with the user's query
        result = graph.invoke({"messages": [("user", query)]}, config)
        return result
    except Exception as e:
        return {"output": str(e)}
```

### Step 6: Set Up Streamlit Frontend

```python
# Create a Streamlit app for the frontend
import streamlit as st
import requests

# Streamlit app title
st.title("Chatbot Interface")

# Input box for user query
user_input = st.text_input("Enter your query:")

# Button to send query
if st.button("Send"):
    try:
        # Send the query to the FastAPI endpoint
        response = requests.get(f"http://localhost:8000/chat/{user_input}")
        
        # Display the response
        if response.status_code == 200:
            st.write("Response:", response.json()["messages"])
        else:
            st.write("Error:", response.text)
    except Exception as e:
        st.write("Error occurred:", str(e))
```

### Step 7: Run the Application

```bash
# Command to run the FastAPI app
poetry run - uvicorn short_term_memory_mongodb_persist:app --reload

# Command to run the Streamlit app
streamlit run streamlit_frontend.py
```

## Summary

This code builds a chatbot that uses:
- LangGraph for managing conversational flow.
- MongoDB for persisting states.
- FastAPI for serving the chatbot via an API.
- Google Gemini AI for generating intelligent responses.
- Streamlit for providing a user-friendly frontend interface.

By following this workflow, the chatbot can handle multi-turn conversations, persist context, and provide a responsive frontend for users.

