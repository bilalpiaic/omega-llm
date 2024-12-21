from dotenv import load_dotenv
load_dotenv()

from enum import Enum
from llama_index.agent.openai import OpenAIAgent

# Import only the functions we need from v1
from v1_basic_agent import (
    get_shared_llm, 
    transformer,
    qa_generator_factory
)

# Agent Types
class Speaker(str, Enum):
    QA_GENERATOR = "Q&A Generator"
    REVIEWER = "Reviewer"

# Reviewer Agent Implementation
def reviewer_factory() -> OpenAIAgent:
    system_prompt = """
    You are the Reviewer agent. Your task is to review and refine Anki flashcards,
    ensuring they follow the minimum information principle.

    Core Review Rules:
    1. Verify each card follows the minimum information principle
    2. Check that Q&A pairs are simple and atomic
    3. Ensure appropriate use of cloze deletions
    4. Verify extra field provides valuable context

    Review Checklist:
    1. Each card should test ONE piece of information
    2. Questions must be:
       - Simple and direct
       - Testing a single fact
       - Using cloze format when appropriate
    3. Answers must be:
       - Brief and precise
       - Limited to essential information
    4. Extra field must include:
       - Detailed explanations
       - Examples
       - Context
    """

    return OpenAIAgent.from_tools(
        [],
        llm=get_shared_llm(),
        system_prompt=system_prompt,
    )

# State Management
def get_initial_state(text: str) -> dict:
    return {
        "input_text": text,
        "qa_cards": "",
        "review_status": "pending"
    }

# Main Function
def generate_anki_cards(input_text: str) -> dict:
    # Initialize state
    state = get_initial_state(input_text)
    
    # Step 1: Generate cards
    generator = qa_generator_factory()
    response = generator.chat(
        f"Generate Anki flashcards from the following text:\n\n{state['input_text']}"
    )
    state["qa_cards"] = str(response)
    print("\nGenerated Cards:")
    print(state["qa_cards"])
    
    # Step 2: Review cards
    reviewer = reviewer_factory()
    review_response = reviewer.chat(
        f"Review and improve these flashcards:\n\n{state['qa_cards']}"
    )
    
    state["qa_cards"] = str(review_response)
    print("\nReviewed Cards:")
    print(state["qa_cards"])
    
    # Transform final cards into structured data
    return transformer(state["qa_cards"])

if __name__ == "__main__":
    sample_text = """
    The Relative Strength Index (RSI) is a momentum indicator used in technical analysis.
    It measures the speed and magnitude of recent price changes to evaluate overbought
    or oversold conditions. RSI is displayed as an oscillator on a scale from 0 to 100.
    Readings above 70 generally indicate overbought conditions, while readings below 30
    indicate oversold conditions.
    """
    
    flashcards = generate_anki_cards(sample_text)
    print("\nFinal Generated and Reviewed Flashcards:")
    print(flashcards)