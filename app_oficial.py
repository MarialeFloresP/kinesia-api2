from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

from fingertap_api import analyze_fingertap

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # para pruebas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):

    # Guardar video temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name

    try:
        results = analyze_fingertap(temp_path)
    except Exception as e:
        os.remove(temp_path)
        return {"error": str(e)}

    os.remove(temp_path)

    return results

