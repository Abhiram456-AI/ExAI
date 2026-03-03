from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference.pipeline import run_pipeline   # adjust if name different

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    crop: str
    treatment: str
    image_path: str
    mode: str

@app.post("/analyze")
def analyze(data: AnalyzeRequest):
    result = run_pipeline(
        data.crop,
        data.treatment,
        data.image_path,
        data.mode
    )

    return result
