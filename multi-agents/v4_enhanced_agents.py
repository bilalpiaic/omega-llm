from pprint import pformat
from typing import List
from dotenv import load_dotenv
load_dotenv()

from enum import Enum
import xml.etree.ElementTree as ET
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.prompts import ChatPromptTemplate 
from llama_index.core.llms import ChatMessage
from tenacity import retry, stop_after_attempt, wait_exponential, TryAgain

# Import base models and functions from previous versions
from v1_basic_agent import Flashcard_model, get_shared_llm, qa_generator_factory
from v2_dual_agents import reviewer_factory
from v3_orchestrated_agents import (
    topic_analyzer_factory, 
    setup_memory
)

# Updated Agent Types
class Speaker(str, Enum):
    QA_GENERATOR = "Q&A Generator"
    REVIEWER = "Reviewer"
    TOPIC_ANALYZER = "Topic Analyzer"
    ORCHESTRATOR = "Orchestrator"
    CODE_AND_EXTRA_FIELD_EXPERT = "Code and Extra Field Expert"
    FORMATTER = "Formatter"

# Code and Extra Field Expert Implementation
def code_and_extra_field_expert_factory() -> OpenAIAgent:
    system_prompt = """
    You are the Code and Extra Field Expert agent. Your task is to enhance Anki flashcards
    by adding relevant code snippets and comprehensive extra content.

    Instructions:
    1. Add clear, concise code examples that illustrate key concepts
    2. Ensure code snippets are well-commented and easy to understand
    3. In the extra field, provide:
       - Step-by-step explanations of code snippets 
       - Common use cases and scenarios
       - Potential pitfalls and edge cases
       - Best practices and optimization tips
    4. Use appropriate markdown formatting for code blocks
    5. Include relevant documentation links
    6. Ensure explanations are clear for a 15-year-old
    """

    return OpenAIAgent.from_tools(
        [],
        llm=get_shared_llm(),
        system_prompt=system_prompt,
    )

# Formatter Agent Implementation
def formatter_agent_factory() -> OpenAIAgent:
    system_prompt = """
    You are the Formatter agent. Your task is to ensure proper XML structure and markdown
    formatting in the flashcards.

    Formatting Rules:
    1. Maintain valid XML structure
    2. Properly escape special characters
    3. Format code blocks with appropriate language tags
    4. Use consistent indentation
    5. Ensure markdown compatibility
    6. Preserve code snippets exactly as provided
    7. Handle nested structures properly
    """

    return OpenAIAgent.from_tools(
        [],
        llm=get_shared_llm(),
        system_prompt=system_prompt,
    )

# Enhanced Error Handling and Validation
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def validate_and_transform(message: str) -> dict:
    try:
        # Transform to structured data
        chat_prompt_tmpl = ChatPromptTemplate(
            message_templates=[
                ChatMessage.from_str(message, role="user")
            ]
        )
        structured_data = get_shared_llm().structured_predict(
            Flashcard_model, 
            chat_prompt_tmpl
        )
        return structured_data.model_dump()
        
    except ET.ParseError as e:
        print(f"XML validation error: {str(e)}")
        raise TryAgain
    except Exception as e:
        print(f"Transformation error: {str(e)}")
        raise TryAgain

# Enhanced State Management
def get_initial_state(text: str) -> dict:
    return {
        "input_text": text,
        "topics": "",
        "qa_cards": "",
        "review_status": "pending",
        "has_code": False,
        "formatting_status": "pending"
    }

# Orchestrator Implementation
def orchestrator_factory(state: dict) -> OpenAIAgent:
    system_prompt = f"""
    You are the Orchestrator agent. Your task is to coordinate the interaction between all agents to create high-quality flashcards.

    Current State:
    {pformat(state, indent=2)}

    Available agents:
    * Topic Analyzer - Breaks down complex topics into structured hierarchical concepts
    * Q&A Generator - Creates flashcards from topics or content
    * Code and Extra Field Expert - Enhances cards with code examples and detailed explanations
    * Reviewer - Reviews and improves card quality, accuracy, and clarity
    * Formatter - Ensures proper XML structure and markdown formatting
    
    Decision Guidelines:
    - Analyze the current state and quality of outputs to decide the next best action
    - You can choose any agent at any time based on need:
        * Use Topic Analyzer when you need better topic understanding or structure, it's the first agent to run
        * Use Q&A Generator when you need new or additional cards
        * Use Code and Extra Field Expert when cards need code examples or detailed explanations
        * Use Reviewer when cards need quality improvement
        * Use Formatter when cards need proper XML/markdown formatting, it's the last agent to run
        * Choose END when the cards are comprehensive and high quality
    
    Examples of flexible decisions:
    - If topic analysis seems incomplete, you can run Topic Analyzer again
    - If cards need improvement, use Reviewer multiple times
    - If cards miss important topics, go back to Q&A Generator
    - If code examples are needed, use Code and Extra Field Expert
    - If formatting needs cleanup, use Formatter
    - If everything looks good, choose END

    Evaluate:
    1. Are the topics well-structured and comprehensive?
    2. Do the cards cover all important concepts?
    3. Are code examples clear and well-explained (if needed)?
    4. Are the cards clear, accurate, and well-written?
    5. Is the formatting correct and consistent?
    6. Is there a good balance of basic and advanced concepts?

    Output only the next agent to run ("Topic Analyzer", "Q&A Generator", "Code and Extra Field Expert", "Reviewer", "Formatter", or "END")
    """

    return OpenAIAgent.from_tools(
        [],
        llm=get_shared_llm(),
        system_prompt=system_prompt,
    )

# Main Function
def generate_anki_cards(input_text: str) -> dict:
    # Initialize state and memory
    state = get_initial_state(input_text)
    memory = setup_memory()
    
    # Detect if input contains code
    state["has_code"] = "```" in input_text or "code" in input_text.lower()
    
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
                analyzer = topic_analyzer_factory()
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
                print("\nGenerated Cards:")
                print(state["qa_cards"])
                
            elif next_agent == Speaker.CODE_AND_EXTRA_FIELD_EXPERT.value:
                expert = code_and_extra_field_expert_factory()
                response = expert.chat(
                    f"Enhance these flashcards with code examples and detailed explanations:\n\n{state['qa_cards']}",
                    chat_history=current_history
                )
                state["qa_cards"] = str(response)
                print("\nEnhanced Cards with Code Examples:")
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
                
            elif next_agent == Speaker.FORMATTER.value:
                formatter = formatter_agent_factory()
                response = formatter.chat(
                    f"Format these flashcards:\n\n{state['qa_cards']}",
                    chat_history=current_history
                )
                state["qa_cards"] = str(response)
                state["formatting_status"] = "completed"
                print("\nFormatted Cards:")
                print(state["qa_cards"])
            
            # Update memory with new interaction
            memory.put(ChatMessage(role="assistant", content=str(response)))
            print(f"\nUpdated memory with {next_agent}'s response")
            
        except Exception as e:
            print(f"\nError in {next_agent}: {str(e)}")
            continue
    
    # Final validation and transformation
    try:
        final_cards = validate_and_transform(state["qa_cards"])
        return final_cards
    except Exception as e:
        print(f"\nError in final transformation: {str(e)}")
        return {}

if __name__ == "__main__":
    sample_text = """
    To calculate the RSI (Relative Strength Index) in Python, you typically use 
    technical analysis libraries like pandas-ta or the ta library. The RSI is 
    calculated using the average gains and losses over a specified period 
    (usually 14 periods). Here's how you can implement it:

    1. Using the ta library:
    ```python
    import pandas as pd
    import ta
    
    # Assuming you have price data in a DataFrame
    df['RSI'] = ta.momentum.RSIIndicator(
        close=df['close'],
        window=14
    ).rsi()
    ```

    2. Manual implementation:
    ```python
    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(
            window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(
            window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    ```
    """
    
    flashcards = generate_anki_cards(sample_text)
    print("Generated Flashcards with Code Examples:")
    print(flashcards)