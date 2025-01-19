from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

# Define a dictionary-like data structure for state using TypedDict
class State(TypedDict):
    input: str  # Represents the input provided by the user
    user_feedback: str  # Represents feedback collected from the user

# Define the first step of the graph
def step_1(state):
    """
    First step in the state graph.
    This can contain logic to process the user's input and make decisions.
    """
    print("---Step 1---")
    pass  # Placeholder for additional processing logic

# Define a step to handle human feedback
def human_feedback(state):
    """
    Function to handle user feedback during an interrupt state.
    Simulates a pause where human feedback is collected.
    """
    print("---human_feedback---")

    # Using `interrupt` to pause execution and collect feedback from the user
    feedback = interrupt({"Please provide feedback:": "WAITING to Start"})

    print("\n\n[GOT BACK FROM HUMAN AFTER INTERRUPT:]\n\n", feedback)
    return {"user_feedback": feedback}

# Define a third step in the graph
def step_3(state):
    """
    Final step in the state graph.
    Represents the completion of the workflow.
    """
    print("---Step 3---")
    pass  # Placeholder for additional logic

# Initialize the StateGraph builder
builder = StateGraph(State)

# Add nodes (steps) to the graph
builder.add_node("step_1", step_1)  # First step
builder.add_node("human_feedback", human_feedback)  # Feedback collection step
builder.add_node("step_3", step_3)  # Final step

# Define edges (transitions) between the nodes
builder.add_edge(START, "step_1")  # Start -> Step 1
builder.add_edge("step_1", "human_feedback")  # Step 1 -> Feedback
builder.add_edge("human_feedback", "step_3")  # Feedback -> Step 3
builder.add_edge("step_3", END)  # Step 3 -> End

# Set up memory for checkpointing
memory = MemorySaver()

# Compile the state graph into an executable form
graph = builder.compile(checkpointer=memory)

# Utility function to check if the agent is in an interrupt state
def is_agent_interrupted(agent, config):
    """
    Checks the state of the agent to see if it's interrupted.
    Returns True if interrupted; otherwise, False.
    """
    # Fetch the current state snapshot
    state_snapshot = agent.get_state(config)

    # Iterate through tasks to check for interrupts
    for task in state_snapshot.tasks:
        if task.interrupts:
            return True
    return False

# Main loop to execute the state graph and handle user input
while True:
    # Example configuration passed to the state graph
    config = {"configurable": {"thread_id": "2"}}
    
    # Accept user input
    user_input = input("Enter your query: ")

    # Invoke the state graph with the user's input and configuration
    graph.invoke({"input": user_input}, config)

    # Check if the agent is in an interrupt state
    is_interrupted = is_agent_interrupted(graph, config)
    
    if is_interrupted:
        print("Agent is interrupted so getting human feedback...")
        
        # Collect human feedback when interrupted
        human_feedback = input("Human Please provide feedback: ")
        
        # Resume the graph from the feedback step and capture the result
        second_result = graph.invoke(Command(resume="human_feedback"), config=config)
        print("second_result", second_result)
