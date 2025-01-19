from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing_extensions import TypedDict
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Initialize FastAPI app
app = FastAPI()

# Key DSA components explained:
# StateGraph: Utilized to define a state machine that guides the process through different stages: creating a post, obtaining human approval, and posting the LinkedIn content.

# Command: Represents the transitions between states in the StateGraph. It contains update data for modifying the state and goto to define the next step.

# interrupt: A method used to simulate human feedback/approval by interrupting the process and receiving approval or disapproval.

# MemorySaver: A class responsible for persisting the state graph, ensuring that previously generated states can be recalled during subsequent executions.

# FastAPI Endpoints:

# /create-post: Receives a topic and initiates the process to generate a LinkedIn post.
# /approve-post: Handles human approval for the generated post by validating the approval response.
# /: A simple root endpoint providing a welcome message.
# Each function in the workflow corresponds to a state in the StateGraph, illustrating the use of a directed graph to manage and transition between different stages of the LinkedIn post creation and publishing process.



# Post ideas dictionary for demonstration
post_ideas = {"yesterday": "on langchain", "today": "on langgraph", "tomorrow": "on langgsmith"}

# Define State TypedDict for LinkedIn Post
class State(TypedDict):
    linkedin_post: str
    is_posted: bool
    topic: str

# Define request body for endpoints
class TopicRequest(BaseModel):
    topic: str  # Topic for which LinkedIn post will be created

class ApprovalRequest(BaseModel):
    approval: str  # "yes" or "no" for human approval

# Function to create LinkedIn post
def create_linkedin_post(state) -> Command[Literal["__end__", "human_approval"]]:
    """
    Creates a LinkedIn post based on the given topic.
    Uses a language model (LLM) to generate content.
    """
    # Initialize ChatGenerativeAI with Google API key and model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    # Invoke LLM to generate LinkedIn post
    linkedin_post = llm.invoke(f"""You are a skilled LinkedIn post writer. Write a post on the topic of {state["topic"]}.""")
    return Command(update={"linkedin_post": linkedin_post}, goto="human_approval")

# Function to post to LinkedIn
def post_to_linkedin(state) -> Command[Literal["__end__"]]:
    """
    Posts the generated LinkedIn content.
    """
    linkedin_post = state["linkedin_post"]
    print(f"Posting to LinkedIn: {linkedin_post}")
    print("Posted successfully!")
    return Command(update={"is_posted": True}, goto="__end__")

# Function to handle human approval
def human_approval(state) -> Command[Literal["__end__", "post_to_linkedin"]]:
    """
    Simulates human feedback and checks whether the post should proceed.
    """
    # Simulate sending for human feedback
    is_approved = interrupt({'task': "Check the post..", 'post': state['linkedin_post']})
    print(f"Human feedback received: {is_approved}")
    if is_approved == "yes":
        return Command(goto="post_to_linkedin")
    else:
        return Command(goto="__end__")

# Set up the StateGraph
builder = StateGraph(State)
builder.add_node("create_linkedin_post", create_linkedin_post)
builder.add_node("human_approval", human_approval)
builder.add_node("post_to_linkedin", post_to_linkedin)
builder.add_edge(START, "create_linkedin_post")

# Memory saver for persistence
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# FastAPI endpoint to handle post creation workflow
@app.post("/create-post")
async def create_post(request: TopicRequest):
    """
    Endpoint to create a LinkedIn post based on the provided topic.
    """
    config = {"configurable": {"thread_id": "2"}}
    result = graph.invoke({"topic": request.topic}, config)
    linkedin_post = result["linkedin_post"].content if "linkedin_post" in result else "No content generated."
    return {"linkedin_post": linkedin_post}

# FastAPI endpoint to handle human approval
@app.post("/approve-post")
async def approve_post(request: ApprovalRequest):
    """
    Endpoint to approve or disapprove the generated LinkedIn post.
    """
    if request.approval not in ["yes", "no"]:
        raise HTTPException(status_code=400, detail="Approval must be 'yes' or 'no'.")
    config = {"configurable": {"thread_id": "2"}}
    graph.invoke(Command(resume=request.approval), config=config)
    return {"status": "Post approved" if request.approval == "yes" else "Post disapproved"}

# FastAPI root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Welcome to the LinkedIn Post Generator API"}
