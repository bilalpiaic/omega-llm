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

# Post ideas dictionary for demonstration
post_ideas = {"yesterday": "on langchain", "today": "on langgraph", "tomorrow": "on langgsmith"}

# Define State TypedDict
class State(TypedDict):
    linkedin_post: str
    is_posted: bool
    topic: str


# Define request body for endpoints
class TopicRequest(BaseModel):
    topic: str


class ApprovalRequest(BaseModel):
    approval: str  # "yes" or "no"


# Function to create LinkedIn post
def create_linkedin_post(state) -> Command[Literal["__end__", "human_approval"]]:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    linkedin_post = llm.invoke(f"""You are a skilled LinkedIn post writer. Write a post on the topic of {state["topic"]}.""")
    return Command(update={"linkedin_post": linkedin_post}, goto="human_approval")


# Function to post to LinkedIn
def post_to_linkedin(state) -> Command[Literal["__end__"]]:
    linkedin_post = state["linkedin_post"]
    print(f"Posting to LinkedIn: {linkedin_post}")
    print("Posted successfully!")
    return Command(update={"is_posted": True}, goto="__end__")


# Function to handle human approval
def human_approval(state) -> Command[Literal["__end__", "post_to_linkedin"]]:
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
    config = {"configurable": {"thread_id": "2"}}
    result = graph.invoke({"topic": request.topic}, config)
    linkedin_post = result["linkedin_post"].content if "linkedin_post" in result else "No content generated."
    return {"linkedin_post": linkedin_post}


# FastAPI endpoint to handle human approval
@app.post("/approve-post")
async def approve_post(request: ApprovalRequest):
    if request.approval not in ["yes", "no"]:
        raise HTTPException(status_code=400, detail="Approval must be 'yes' or 'no'.")
    config = {"configurable": {"thread_id": "2"}}
    graph.invoke(Command(resume=request.approval), config=config)
    return {"status": "Post approved" if request.approval == "yes" else "Post disapproved"}


# FastAPI root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Welcome to the LinkedIn Post Generator API"}
