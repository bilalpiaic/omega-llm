# Create a Streamlit app for the frontend
import streamlit as st
import requests

# Streamlit app title
st.title("Chatbot Interface")

# Input box for user query
user_input = st.text_input("Enter your query:")

# Button to send query
if st.button("Send"):
    try:
        # Send the query to the FastAPI endpoint
        response = requests.get(f"http://localhost:8000/chat/{user_input}")
        
        # Display the response
        if response.status_code == 200:
            st.write("Response:", response.json()["messages"])
        else:
            st.write("Error:", response.text)
    except Exception as e:
        st.write("Error occurred:", str(e))