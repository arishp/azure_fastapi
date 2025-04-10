from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI()

class LlmResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Azure!"}

@app.get("/query", response_model=LlmResponse)
async def query_llm(request: str):
    try:
        load_dotenv()
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        response = llm.invoke(request)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))