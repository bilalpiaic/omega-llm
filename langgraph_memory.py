# from typing_extensions import TypedDict
# from typing import Annotated
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_google_genai import ChatGoogleGenerativeAI
# from fastapi import FastAPI



# from dotenv import load_dotenv
# import os
# load_dotenv()

# llm = ChatGoogleGenerativeAI( 
#     model="gemini-1.5-flash", 
#     google_api_key=os.getenv("GOOGLE_API_KEY"))


# class MessagesState(TypedDict):
#     messages: Annotated[list, add_messages]
    


# def assistant(state: MessagesState):
#     return {"messages": [llm.invoke(state["messages"])]}

# # Build graph
# builder = StateGraph(MessagesState)
# builder.add_node("assistant", assistant)
# builder.add_edge(START, "assistant")
# builder.add_edge("assistant", END)
# checkpointer = MemorySaver()
# graph = builder.compile(checkpointer=checkpointer)


# app = FastAPI()

# @app.get("/chat/{query}")
# def get_content(query: str):
#     print(query)
#     try:
#         config = {"configurable": {"thread_id": "1"}}
#         result = graph.invoke({"messages": [("user", query)]}, config)
#         return result
#     except Exception as e:
#         return {"output": str(e)}

from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]

def assistant(state: MessagesState):
    """Assistant node to process the user's messages and generate AI responses."""
    user_input = state["messages"][-1][1]  # Get the latest user query
    ai_response = llm.invoke(state["messages"])
    state["messages"].append(("assistant", ai_response))  # Append AI response
    return {"messages": state["messages"]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", "assistant")
builder.add_edge("assistant", END)
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Initialize FastAPI app
app = FastAPI()

@app.get("/chat/{query}")
def get_content(query: str):
    """API endpoint to handle user queries and generate 5 AI responses."""
    print(query)
    try:
        config = {"configurable": {"thread_id": "1"}}
        # Initialize the conversation with the user's first query
        conversation_state = {"messages": [("user", query)]}

        # Process 5 user/AI exchanges
        for _ in range(5):
            conversation_state = graph.invoke(conversation_state, config)

        # Return the final conversation history
        return conversation_state
    except Exception as e:
        return {"output": str(e)}
    



# poetry run python -m uvicorn langgraph_memory:app --reload