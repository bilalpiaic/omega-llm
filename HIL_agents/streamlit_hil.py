import streamlit as st
import requests

# Backend API URL
BASE_URL = "http://127.0.0.1:8000"  # Replace with your FastAPI backend URL if hosted elsewhere

st.title("LinkedIn Post Generator")
st.subheader("Create and review LinkedIn posts with AI assistance!")

# Step 1: Enter the topic
st.write("### Step 1: Enter a Topic for the LinkedIn Post")
topic = st.text_input("Enter your topic:", "")

if st.button("Generate Post"):
    if not topic.strip():
        st.error("Please enter a topic before generating the post.")
    else:
        with st.spinner("Generating LinkedIn post..."):
            response = requests.post(f"{BASE_URL}/create-post", json={"topic": topic})
            if response.status_code == 200:
                linkedin_post = response.json().get("linkedin_post", "No content generated.")
                st.session_state["linkedin_post"] = linkedin_post
                st.success("Post generated successfully!")
                st.write("### Generated LinkedIn Post:")
                st.write(linkedin_post)
            else:
                st.error(f"Error: {response.json().get('detail', 'Something went wrong')}")

# Step 2: Human approval
if "linkedin_post" in st.session_state:
    st.write("### Step 2: Approve or Disapprove the Post")
    st.write("Generated Post:")
    st.write(st.session_state["linkedin_post"])

    approval = st.radio("Do you approve this post?", ["Yes", "No"], key="approval_choice")
    if st.button("Submit Approval"):
        with st.spinner("Submitting your approval..."):
            response = requests.post(f"{BASE_URL}/approve-post", json={"approval": approval.lower()})
            if response.status_code == 200:
                status = response.json().get("status", "Approval process completed.")
                if approval == "Yes":
                    st.success(f"✅ {status}")
                else:
                    st.warning(f"❌ {status}")
            else:
                st.error(f"Error: {response.json().get('detail', 'Something went wrong')}")

# Footer
st.write("---")
st.write("Created with ❤️ by Streamlit and FastAPI")
