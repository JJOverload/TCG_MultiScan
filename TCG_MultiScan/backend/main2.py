import uvicorn #comment if not using optional line
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def index():
    return {"message": "Hello World!"}


@app.get("/hello/{name}")
async def hello(name):
   return {"name": name}

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