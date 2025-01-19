from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import os
from langgraph.types import Command, interrupt

# Predefined topics for LinkedIn posts for demonstration purposes
post_ideas = {"yesterday": "on langchain", "today": "on langgraph", "tommorow": "on langsmith"}

# Define the state structure to manage data throughout the workflow
class State(TypedDict):
    linkedin_post: str  # The content of the LinkedIn post
    is_posted: bool  # Status indicating if the post is successfully published
    topic: str  # The topic for the LinkedIn post

# Step 1: Create a LinkedIn post using Google Generative AI
def create_linkedin_post(state) -> Command[Literal["__end__", "human_approval"]]:
    """
    This function generates a LinkedIn post using an AI model.
    It takes the topic from the state and invokes the AI to create a post.
    """
    # Initialize the AI model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

    # Use the model to generate a LinkedIn post
    linkedin_post = llm.invoke(f"""You are a skilled LinkedIn post writer. write a post on the topic of {state["topic"]}""")
    
    # Return the generated post and move to the human approval step
    return Command(update={"linkedin_post": linkedin_post}, goto="human_approval")

# Step 2: Post to LinkedIn
def post_to_linkedin(state) -> Command[Literal["__end__"]]:
    """
    This function simulates posting the content to LinkedIn.
    It updates the state to indicate the post was successfully published.
    """
    linkedin_post = state["linkedin_post"]
    print(f"Posting to LinkedIn...")
    print("Posted")
    # Update the state and mark the process as finished
    return Command(update={"is_posted": True}, goto="__end__")

# Step 3: Human approval process
def human_approval(state) -> Command[Literal["__end__", "post_to_linkedin"]]:
    """
    This function handles human feedback for the generated LinkedIn post.
    An interrupt is used to collect the feedback interactively.
    """
    print("---human_feedback---")

    # Interrupt to ask for human approval of the generated post
    is_approved = interrupt({'task': "Check the post..", 'post': state['linkedin_post']})

    print("\n\n[RESUME AFTER INTERRUPT:]\n\n", is_approved)

    # If approved, move to the posting step; otherwise, terminate the workflow
    if is_approved == "yes":
        return Command(goto="post_to_linkedin")
    else:
        return Command(goto="__end__")

# Initialize the state graph
builder = StateGraph(State)

# Add nodes (steps) to the graph
builder.add_node("create_linkedin_post", create_linkedin_post)  # Step 1
builder.add_node("human_approval", human_approval)  # Step 2: Feedback
builder.add_node("post_to_linkedin", post_to_linkedin)  # Step 3: Post

# Define the workflow edges
builder.add_edge(START, "create_linkedin_post")  # Start -> Create Post

# Set up memory for checkpointing
memory = MemorySaver()

# Compile the graph into an executable form
graph = builder.compile(checkpointer=memory)

# Utility function to check if the agent is in an interrupt state
def is_agent_interrupted(agent, config):
    """
    Checks the agent's state for any pending interruptions.
    Returns True if interrupted, False otherwise.
    """
    # Get the current state snapshot
    state_snapshot = agent.get_state(config)

    # Iterate through tasks to check for interrupts
    for task in state_snapshot.tasks:
        if task.interrupts:
            return True
    return False

# Main loop to execute the state graph
while True:
    # Configuration passed to the state graph
    config = {"configurable": {"thread_id": "2"}}
    
    # Accept user input for the LinkedIn post topic
    user_input = input("Enter the topic: ")

    # Invoke the state graph with the input topic
    result = graph.invoke({"topic": user_input}, config)
    
    # Output the generated LinkedIn post
    print('result:', result['linkedin_post'].content)
    
    # Check if the agent is interrupted (awaiting human feedback)
    is_interrupted = is_agent_interrupted(graph, config)
    
    if is_interrupted:
        print("Agent is interrupted so getting human feedback...")
        
        # Collect human feedback for the generated post
        human_feedback = input("Your approval is required. Read the post and say 'yes' to approve or 'no' to disapprove: ")
        
        # Resume the graph based on the human feedback
        graph.invoke(Command(resume=human_feedback), config=config)
