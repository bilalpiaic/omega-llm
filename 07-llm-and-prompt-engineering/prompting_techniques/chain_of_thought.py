# Chain-of-Thought (CoT) Prompting is a technique in prompt engineering where a model is guided to solve a problem by breaking it down into intermediate reasoning steps. Instead of directly generating the final answer, the model is prompted to "think aloud" by producing a sequence of logical steps that lead to the solution. This approach mimics human problem-solving and is particularly useful for complex tasks like math problems, logical reasoning, or multi-step decision-making. By explicitly showing the reasoning process, CoT prompting improves the model's accuracy, transparency, and ability to handle challenging problems.

from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Define the Chain-of-Thought prompt
cot_prompt = """
Solve the following math problem step by step:

Problem: A farmer has 10 chickens and 5 cows. If each chicken lays 2 eggs per day and each cow produces 3 liters of milk per day, how many eggs and liters of milk does the farmer get in 7 days?

"""

# Invoke the LLM with the Chain-of-Thought prompt
result = llm.invoke(cot_prompt)

# Print the result
print(result)