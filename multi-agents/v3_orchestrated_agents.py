from typing import List
from dotenv import load_dotenv
load_dotenv()

from enum import Enum
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.prompts import ChatPromptTemplate 
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from pprint import pformat

# Import base functions from previous versions
from v1_basic_agent import (
    get_shared_llm,
    transformer,
    qa_generator_factory
)
from v2_dual_agents import reviewer_factory

# Agent Types
class Speaker(str, Enum):
    QA_GENERATOR = "Q&A Generator"
    REVIEWER = "Reviewer"
    TOPIC_ANALYZER = "Topic Analyzer"
    ORCHESTRATOR = "Orchestrator"

# Topic Analyzer Implementation
def topic_analyzer_factory(state: dict) -> OpenAIAgent:
    system_prompt = f"""
    You are the Topic Analyzer agent. Your task is to analyze the given text and identify key topics for flashcard creation.
    
    Current State:
    {pformat(state, indent=2)}
    
    Instructions:
    1. Identify main concepts, sub-concepts, and their relationships
    2. Create a hierarchical structure of topics
    3. Highlight potential areas for deep-dive questions
    4. Identify cross-cutting themes or principles
    5. Suggest real-world applications or case studies

    Output format:
    <topics>
        <topic>
            <name>Main topic name</name>
            <subtopics>
                <subtopic>Subtopic 1</subtopic>
                <subtopic>Subtopic 2</subtopic>
            </subtopics>
            <applications>Potential real-world applications</applications>
            <prerequisites>Foundational knowledge needed</prerequisites>
        </topic>
    </topics>
    """

    return OpenAIAgent.from_tools(
        [],
        llm=get_shared_llm(),
        system_prompt=system_prompt,
        verbose=True
    )

# Orchestrator Implementation
def orchestrator_factory(state: dict) -> OpenAIAgent:
    system_prompt = f"""
    You are the Orchestrator agent. Your task is to coordinate the interaction between all agents to create high-quality flashcards.

    Current State:
    {pformat(state, indent=2)}

    Available agents:
    * Topic Analyzer - Breaks down complex topics into structured hierarchical concepts
    * Q&A Generator - Creates flashcards from topics or content
    * Reviewer - Reviews and improves card quality, accuracy, and clarity
    
    Decision Guidelines:
    - Analyze the current state and quality of outputs to decide the next best action
    - You can choose any agent at any time based on need:
        * Use Topic Analyzer when you need better topic understanding or structure, it's the first agent to run
        * Use Q&A Generator when you need new or additional cards
        * Use Reviewer when cards need quality improvement
        * Choose END when the cards are comprehensive and high quality
    
    Examples of flexible decisions:
    - If topic analysis seems incomplete, you can run Topic Analyzer again
    - If cards need improvement, use Reviewer multiple times
    - If cards miss important topics, go back to Q&A Generator
    - If everything looks good, choose END

    Evaluate:
    1. Are the topics well-structured and comprehensive?
    2. Do the cards cover all important concepts?
    3. Are the cards clear, accurate, and well-written?
    4. Is there a good balance of basic and advanced concepts?

    Output only the next agent to run ("Topic Analyzer", "Q&A Generator", "Reviewer", or "END")
    """

    return OpenAIAgent.from_tools(
        [],
        llm=get_shared_llm(),
        system_prompt=system_prompt,
    )

# Memory Management
def setup_memory() -> ChatMemoryBuffer:
    return ChatMemoryBuffer.from_defaults(token_limit=8000)

# Enhanced State Management
def get_initial_state(text: str) -> dict:
    return {
        "input_text": text,
        "topics": "",
        "qa_cards": "",
        "review_status": "pending",
    }

# Main Function
def generate_anki_cards(input_text: str) -> dict:
    # Initialize state and memory
    state = get_initial_state(input_text)
    memory = setup_memory()
    
    while True:
        # Get current chat history
        current_history = memory.get()
        
        # Let Orchestrator decide next step
        orchestrator = orchestrator_factory(state)
        next_agent = str(orchestrator.chat(
            "Decide which agent to run next based on the current state.",
            chat_history=current_history
        )).strip().strip('"').strip("'")
        print(f"\nOrchestrator selected: {next_agent}")
        
        if next_agent == "END":
            print("\nOrchestrator decided to end the process")
            break
            
        # Execute selected agent
        try:
            if next_agent == Speaker.TOPIC_ANALYZER.value:
                analyzer = topic_analyzer_factory(state)
                response = analyzer.chat(
                    f"Analyze this text for flashcard topics:\n\n{state['input_text']}",
                    chat_history=current_history
                )
                state["topics"] = str(response)
                print("\nTopic Analysis Results:")
                print(state["topics"])
                
            elif next_agent == Speaker.QA_GENERATOR.value:
                generator = qa_generator_factory()
                response = generator.chat(
                    f"Generate flashcards for this topic:\n\n{state['topics']}",
                    chat_history=current_history
                )
                state["qa_cards"] = str(response)
                state["review_status"] = "needs_review"
                print("\nGenerated Cards:")
                print(state["qa_cards"])
                
            elif next_agent == Speaker.REVIEWER.value:
                reviewer = reviewer_factory()
                response = reviewer.chat(
                    f"Review these flashcards:\n\n{state['qa_cards']}",
                    chat_history=current_history
                )
                state["qa_cards"] = str(response)
                state["review_status"] = "reviewed"
                print("\nReviewed Cards:")
                print(state["qa_cards"])
                
                # Update memory with new interaction
                memory.put(ChatMessage(role="assistant", content=str(response)))
                print(f"\nUpdated memory with {next_agent}'s response")
            
        except Exception as e:
            print(f"\nError in {next_agent}: {str(e)}")
            continue
        
    # Transform final cards into structured data
    final_cards = transformer(state["qa_cards"])
    return final_cards

if __name__ == "__main__":
    sample_text = """
    The Relative Strength Index (RSI) is a momentum indicator used in technical analysis.
    It measures the speed and magnitude of recent price changes to evaluate overbought
    or oversold conditions. RSI is displayed as an oscillator on a scale from 0 to 100.
    Readings above 70 generally indicate overbought conditions, while readings below 30
    indicate oversold conditions. The RSI can also help identify divergences, trend lines,
    and failure swings that may not be apparent on the underlying price chart.
    """
    
    flashcards = generate_anki_cards(sample_text)
    print("\nGenerated, Analyzed, and Reviewed Flashcards:")
    print(flashcards)