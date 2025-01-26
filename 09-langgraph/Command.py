import random
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# Define the state structure to store query and response
class State(TypedDict):
    query: str       # User's input query
    response: str    # Response generated based on the query


# Function to handle booking-related queries
def handle_booking(state: State) -> Command[Literal["__end__"]]:
    """
    Processes booking requests.
    Updates the state with a booking response and directs the workflow to END.
    """
    state["response"] = "Your booking request is being processed."
    return Command(update=state, goto="__end__")  # Update the state and move to END


# Function to handle cancellation-related queries
def handle_cancellation(state: State) -> Command[Literal["__end__"]]:
    """
    Processes cancellation requests.
    Updates the state with a cancellation response and directs the workflow to END.
    """
    state["response"] = "Your cancellation request is being processed."
    return Command(update=state, goto="__end__")


# Function to handle general information requests
def handle_general_info(state: State) -> Command[Literal["__end__"]]:
    """
    Provides general information for queries unrelated to booking or cancellation.
    Updates the state with a general info response and directs the workflow to END.
    """
    state["response"] = "Here is some general information about our services."
    return Command(update=state, goto="__end__")


# Function to route the query to the appropriate handler
def route_query(state: State) -> Command[Literal["booking", "cancellation", "general_info"]]:
    """
    Routes the query based on its content:
    - Routes to 'booking' if the query contains 'book'.
    - Routes to 'cancellation' if the query contains 'cancel'.
    - Defaults to 'general_info' for other queries.
    """
    if "book" in state["query"].lower():
        return Command(goto="booking")  # Route to booking handler
    elif "cancel" in state["query"].lower():
        return Command(goto="cancellation")  # Route to cancellation handler
    else:
        return Command(goto="general_info")  # Route to general info handler


# Create a state graph for the workflow
graph = StateGraph(State)

## Previously defined nodes and edges
# graph.add_node("route_query", route_query)
# graph.add_node("booking", handle_booking)
# graph.add_node("cancellation", handle_cancellation)
# graph.add_node("general_info", handle_general_info)

# # Define the edges
# graph.add_edge(START, "route_query")
# graph.add_conditional_edges("route_query", {
#     "booking": "booking",
#     "cancellation": "cancellation",
#     "general_info": "general_info"
# })
# graph.add_edge("booking", END)
# graph.add_edge("cancellation", END)
# graph.add_edge("general_info", END)



# Add nodes (functions) to the graph
graph.add_node("route_query", route_query)  # Initial query routing
graph.add_node("booking", handle_booking)  # Handle booking
graph.add_node("cancellation", handle_cancellation)  # Handle cancellation
graph.add_node("general_info", handle_general_info)  # Handle general info

# Define the transition from the START node to the query routing node
graph.add_edge(START, "route_query")

# Compile the graph into an executable app
app = graph.compile()

# Main execution loop to handle user input
while True:
    # Get user input
    user_input = input("Enter your query: ")
    
    # Invoke the graph with the user's input and store the result
    result = app.invoke({"query": user_input})
    
    # Output the response from the state
    print(result)