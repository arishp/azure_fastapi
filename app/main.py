from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

import logging
import sys

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from app.graph import builder
import uuid 


logger = logging.getLogger("uvicorn")  # Ensures compatibility with Uvicorn logging
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


app = FastAPI()


class LlmResponse(BaseModel):
    response: str


@app.get("/")
def read_root():
    logger.info("Hello from FastAPI root endpoint")
    return {"message": "Hello from FastAPI on Azure!"}


@app.get("/query", response_model=LlmResponse)
async def query_llm(request: str):
    logger.info("Hello from FastAPI query endpoint")
    try:
        load_dotenv()
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        response = llm.invoke(request)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dr", response_model=LlmResponse)
async def deep_research(request: str):
    logger.info("Hello from FastAPI dr endpoint")
    REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

    1. Introduction
    - Brief overview of the topic area

    2. Main Body Sections:
    - Each section should focus on a sub-topic of the user-provided topic
    
    3. Conclusion
    - Aim for 1 structural element (either a list of table) that distills the main body sections 
    - Provide a concise summary of the report"""
    try:
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)
        load_dotenv()
        thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                                "search_api": "tavily",
                                "planner_provider": "google-genai",
                                "planner_model": "gemini-2.0-flash-lite",
                                "writer_provider": "google-genai",
                                "writer_model": "gemini-2.0-flash-lite",
                                "max_search_depth": 1,
                                "report_structure": REPORT_STRUCTURE,
                                }}
        async for event in graph.astream({"topic":request,}, thread, stream_mode="updates"):
            if '__interrupt__' in event:
                logger.info("Created report structure")
        # passing feedback
        # async for event in graph.astream(Command(resume="Include individuals sections for each topic"), thread, stream_mode="updates"):
        #     pass
        async for event in graph.astream(Command(resume=True), thread, stream_mode="updates"):
            logger.info(list(event.keys())[0])
        final_state = graph.get_state(thread)
        report = final_state.values.get('final_report')
        return {"response": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))