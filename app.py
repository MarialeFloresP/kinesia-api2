from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

from fingertap_api import analyze_fingertap
from opening_api import analyze_opening
from pronation_api import analyze_pronation

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
import json

# Appwrite imports
from appwrite.client import Client
from appwrite.services.storage import Storage
from appwrite.input_file import InputFile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # para pruebas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURACIÓN DE APPWRITE ---
client = Client()
client.set_endpoint(os.environ.get("APPWRITE_ENDPOINT")) 
client.set_project(os.environ.get("APPWRITE_PROJECT_ID")) 
client.set_key(os.environ.get("APPWRITE_API_KEY")) 

storage = Storage(client)
BUCKET_ID = "videos_bucket"


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
            os.remove(temp_path)
            return {"error": "Invalid movement type"}
            
        
        print("Análisis completado")
    
    except Exception as e:
        os.remove(temp_path)
        return {"error": str(e)}
    
    os.remove(temp_path)

    return results

# Endpoint para subir videos  -------------------------
@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        print("Subiendo video a Appwrite:", file.filename)

        # Leer contenido
        file_bytes = await file.read()
        
        # Generar un ID único para el archivo
        file_id = str(uuid.uuid4())[:20]

        # Subir a Appwrite
        result = storage.create_file(
            bucket_id=BUCKET_ID,
            file_id=file_id,
            file=InputFile.from_bytes(file_bytes, filename=file.filename)
        )

        # URL para ver/descargar el video
        # Nota: Asegúrate de que los permisos del bucket estén en "Any" para Read
        video_url = f"{os.environ.get('APPWRITE_ENDPOINT')}/storage/buckets/{BUCKET_ID}/files/{file_id}/view?project={os.environ.get('APPWRITE_PROJECT_ID')}"

        print(f"Éxito! Archivo subido a Appwrite con ID: {file_id}")

        return {
            "fileId": file_id, 
            "status": "success",
            "url": video_url
        }

    except Exception as e:
        print("ERROR EN APPWRITE:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
