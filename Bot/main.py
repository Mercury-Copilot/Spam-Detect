#import required libraries
from fastapi import FastAPI, HTTPException
import os
import google.generativeai as genai
from dotenv import load_dotenv

#loading environment variable
load_dotenv('api_key.env')

api_key = os.environ.get('GENAI_API_KEY')

#configuring api key
genai.configure(api_key=api_key)

#loading fine tuned model
model = genai.GenerativeModel(model_name='tunedModels/mercury-w0aigfuv20g3')

#defining function generate responses
def generate_response(prompt):
    response = model.generate_content(prompt)
    return response.text

#configure application
app = FastAPI()

#root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Mercury API!"}

#endpoint
@app.post("/generate/")
@app.get("/generate/")
async def generate_content(query):
    try:
        return generate_response(query)
    except Exception as e:
        raise HTTPException(status_code = 500, detail = 'Failed to generate response.')