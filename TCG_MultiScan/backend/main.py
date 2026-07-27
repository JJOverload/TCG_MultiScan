from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow connections from your React development server
origins = [
    "http://localhost:5173", # Default Vite port
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/data")
async def get_data():
    return {"message": "Hello from the FastAPI backend!"}


# To start your backend server locally using Uvicorn
# uvicorn main:app --reload

# Use Vite to scaffold a fast modern React environment
# npm create vite@latest frontend -- --template react
# cd frontend
# npm install