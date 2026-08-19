import uvicorn #comment if not using optional line
from fastapi import FastAPI
#for CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# --------------------------------------
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
# --------------------------------------


# Check using "http://localhost:8000"... More details on the bottom.
@app.get("/")
async def index():
    return {"message": "Hello World!"}


@app.get("/hello/{name}")
async def hello(name):
   return {"name": name}

@app.get("/hello/{name}/{age}")
async def hello(name:str, age:int):
    return {"name": name, "age": age}

# optional
if __name__ == "__main__":
    uvicorn.run("main2:app", host="127.0.0.1", port=8000, reload=True)


"""
Optional code replaces the need to run:

uvicorn main2:app --reload

each time.
"""
"""
Start the Uvicorn server and visit http://localhost:8000/hello/Tutorialspoint URL. The browser shows the following JSON response.

{"name":"Tutorialspoint"}
"""

"""
Change the variable path parameter to something else such as http://localhost:8000/hello/Python so that the browser shows −

{"name":"Python"}
"""