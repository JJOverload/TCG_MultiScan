# README for Whole Frontend + Backend Setup for TCG_MultiScan


## Table of Contents
* [Set up FastAPI Backend](#set-up-fastapi-backend)
  * [If Backend Already Configured](#if-backend-already-configured)
* [Set up React Frontend](#set-up-react-frontend)
* [References](#references)



## Set up FastAPI Backend

First create .venv using python:
~~~
mkdir backend && cd backend
python -m venv .venv
# Activate environment:
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
~~~

Then make sure requirements.txt has this inside:
~~~
fastapi
uvicorn
~~~

Install them using pip: `pip install -r requirements.txt`

(Can skip this section if you already have virtual environment setup for this project.)

------------------------------------

## If Backend Already Configured

Ensure that the main.py (from the backend folder) is present.

Then, to start your backend server locally using Uvicorn:
`uvicorn main:app --reload`

------------------------------------

## Set up React Frontend

Use Vite to scaffold a fast modern React environment:
~~~
npm create vite@latest frontend -- --template react
cd frontend
npm install
~~~

If scaffolding done, can go straight to starting your React development environment:
`npm run dev`

(Note to self: for production, would want to use slightly different commands for presentation under real-world settings.)

Open http://localhost:5173 in your web browser to see your React interface successfully 
displaying data parsed straight from your backend


------------------------------------

`pip install mypy`

------------------------------------




## References

Google AI mode for generating instructions for initial setup and template code:
https://share.google/aimode/WarGJK0N47lRro2J5