from typing import List
from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.prompts import ChatPromptTemplate 
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

# Data Models
class QACard(BaseModel):
    question: str
    answer: str
    extra: str

class Flashcard_model(BaseModel):
    cards: List[QACard]

# LLM Configuration
def get_shared_llm():
    """Returns a shared LLM instance for all agents."""
    return OpenAI(model="gpt-4o-mini", temperature=0, api_base="http://127.0.0.1:4000/v1", api_key="sk-test")

# Basic Agent Implementation
def qa_generator_factory() -> OpenAIAgent:
    system_prompt = """
    You are an educational content creator specializing in Anki flashcard generation.
    Your task is to create clear, concise flashcards following these guidelines:

    1. Each card should focus on ONE specific concept
    2. Questions should be clear and unambiguous
    3. Answers should be concise but complete
    4. Include relevant extra information in the extra field
    5. Follow the minimum information principle

    Format each card as:
    <card>
        <question>Your question here</question>
        <answer>Your answer here</answer>
        <extra>Additional context, examples, or explanations</extra>
    </card>
    """

    return OpenAIAgent.from_tools(
        [],
        llm=get_shared_llm(),
        system_prompt=system_prompt,
    )

# Transformer Function
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def transformer(message: str) -> dict:
    chat_prompt_tmpl = ChatPromptTemplate(
        message_templates=[
            ChatMessage.from_str(message, role="user")
        ]
    )
    llm = get_shared_llm()
    structured_data = llm.structured_predict(Flashcard_model, chat_prompt_tmpl)
    return structured_data.model_dump()

# Main Function
def generate_anki_cards(input_text: str) -> dict:
    # Create the agent
    agent = qa_generator_factory()
    
    # Generate flashcards
    response = agent.chat(
        f"Generate Anki flashcards from the following text:\n\n{input_text}"
    )
    print("Raw Agent Response:")
    print(response)
    
    # Transform the response into structured data
    structured_response = transformer(str(response))
    print("\nStructured Response:")
    print(structured_response)
    
    return structured_response

if __name__ == "__main__":
    sample_text = """
    The Relative Strength Index (RSI) is a momentum indicator used in technical analysis.
    It measures the speed and magnitude of recent price changes to evaluate overbought
    or oversold conditions. RSI is displayed as an oscillator on a scale from 0 to 100.
    Readings above 70 generally indicate overbought conditions, while readings below 30
    indicate oversold conditions.
    """
    
    flashcards = generate_anki_cards(sample_text)
    print("Generated Flashcards:")
    print(flashcards)