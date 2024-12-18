from fastapi import FastAPI

# Create FastAPI application instance
app = FastAPI()

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}

# Additional endpoint example
@app.get("/hello/{name}")
def say_hello(name: str):
    return {"message": f"Hello, {name}! Welcome to the FastAPI application."}

#poetry run python -m uvicorn simple_fastapi:app --reload