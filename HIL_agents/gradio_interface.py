import gradio as gr
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import time


def generate_post_with_streaming(topic: str):
    """
    Simulates streaming by generating the LinkedIn post character by character.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    linkedin_post = llm.invoke(f"""You are a skilled LinkedIn post writer. Write a post on the topic of {topic}.""")
    
    # Stream content character by character
    for i in range(1, len(linkedin_post) + 1):
        yield f"### LinkedIn Post:\n{linkedin_post[:i]}"
        time.sleep(0.08)  # Delay for streaming effect


def approve_post(post: str, approval: str) -> str:
    """
    Simulates approval/disapproval of the LinkedIn post.
    """
    if approval not in ["yes", "no"]:
        return "Approval must be 'yes' or 'no'."
    return f"**Post Status:** {'Approved ✅' if approval == 'yes' else 'Disapproved ❌'}"


# Gradio Interface
gr.Interface(
    fn=lambda topic, approval: (
        generate_post_with_streaming(topic),
        approve_post("Simulated Post Content", approval),
    ),
    inputs=[
        gr.components.Textbox(label="Topic"),
        gr.components.Radio(choices=["yes", "no"], label="Approve Post?")
    ],
    outputs=[
        gr.components.Markdown(label="Generated LinkedIn Post", stream=True),
        gr.components.Markdown(label="Approval Status")
    ],
    title="LinkedIn Post Generator",
    description="Generate and approve LinkedIn posts using AI with a streaming effect."
).launch()
