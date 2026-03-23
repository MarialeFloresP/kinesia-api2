from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

from fingertap_api import analyze_fingertap
from opening_api import analyze_opening
from pronation_api import analyze_pronation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # para pruebas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...), movement: str = Form(...)):
    
    print("Video recibido:", file.filename)
    
    # Guardar video temporal por chunks (NO carga todo en RAM)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        temp_path = tmp.name
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB por vez
            if not chunk:
                break
            tmp.write(chunk)

    try:
        if movement == "fingertap":
            results = analyze_fingertap(temp_path)
        elif movement == "opening":
            results = analyze_opening(temp_path)
        elif movement == "pronation":
            results = analyze_pronation(temp_path)
        else:
            return {"error": "Invalid movement type"}
        
        print("Análisis completado")
    
    except Exception as e:
        os.remove(temp_path)
        return {"error": str(e)}

    os.remove(temp_path)


    return results
