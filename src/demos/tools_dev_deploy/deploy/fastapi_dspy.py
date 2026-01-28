from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import dspy

'''
启动服务：
uvicorn fastapi_dspy:app --reload
访问服务：
http://127.0.0.1:8000/
'''

app = FastAPI(
    title="DSPy Program API",
    description="A simple API serving a DSPy Chain of Thought program",
    version="1.0.0"
)

# Define request model for better documentation and validation
class Question(BaseModel):
    text: str

# Configure your language model and 'asyncify' your DSPy program.
lm = dspy.LM(model="openai/llama3.2:3b",
             api_key="sk-1234567890abcdef1234567890abcdef",
             api_base="https://localhost:11111/v1",
             )
dspy.configure(lm=lm, async_max_workers=4) # default is 8
dspy_program = dspy.ChainOfThought("question -> answer")
dspy_program = dspy.asyncify(dspy_program)

@app.post("/predict")
async def predict(question: Question):
    try:
        result = await dspy_program(question=question.text)
        return {
            "status": "success",
            "data": result.toDict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))