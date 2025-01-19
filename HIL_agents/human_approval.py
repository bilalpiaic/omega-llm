from typing_extensions import TypedDict
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

# Define the state structure for the workflow
class State(TypedDict):
    input: str  # Input query provided by the user

# Step 1: Human approval process
def human_approval(state) -> Command[Literal["__end__", "call_agent"]]:
    """
    This function simulates a human feedback step.
    It uses an interrupt to pause execution and ask for human approval.
    """
    print("---human_feedback---")
    
    # Interrupt the workflow to collect human approval
    is_approved = interrupt("Is this correct?")

    print("\n\n[RESUME AFTER INTERRUPT:]\n\n", is_approved)

    # If approved, move to the "call_agent" step; otherwise, terminate the workflow
    if is_approved == "yes":
        return Command(goto="call_agent")
    else:
        return Command(goto="__end__")

# Step 2: Call the agent
def call_agent(state):
    """
    This function represents the next step after human approval.
    It can execute additional logic or interact with an external agent/system.
    """
    print("---call_agent 3---")
    pass

# Initialize the state graph
builder = StateGraph(State)

# Add nodes (steps) to the graph
builder.add_node("human_approval", human_approval)  # Step 1: Human feedback
builder.add_node("call_agent", call_agent)  # Step 2: Call agent

# Define the workflow edges
builder.add_edge(START, "human_approval")  # Start -> Human approval

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

# Main execution loop
while True:
    # Configuration for the state graph
    config = {"configurable": {"thread_id": "2"}}
    
    # Accept user input for the workflow
    user_input = input("Enter your query: ")
    
    # Invoke the graph with the user input
    graph.invoke({"input": user_input}, config)
    
    # Check if the agent is interrupted
    is_interrupted = is_agent_interrupted(graph, config)
    
    if is_interrupted:
        print("Agent is interrupted so getting human feedback...")
        
        # Collect human feedback interactively
        human_feedback = input("Human Please provide feedback: ")
        
        # Resume the graph with the human feedback
        second_result = graph.invoke(Command(resume=human_feedback), config=config)
        print("second_result", second_result)
