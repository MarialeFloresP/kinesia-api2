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
            os.remove(temp_path)
            return {"error": "Invalid movement type"}
            
        
        print("Análisis completado")
    
    except Exception as e:
        os.remove(temp_path)
        return {"error": str(e)}
    
    os.remove(temp_path)

    return results

# Endpoint para subir videos a Drive -------------------------

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        print("Subiendo video a Drive:", file.filename)

        # 1. Ajustamos el nombre de la variable de entorno a KEY_JSON (como lo pusiste en Render)
        raw_json = os.environ.get("KEY_JSON")
        if not raw_json:
            raise HTTPException(status_code=500, detail="La variable KEY_JSON no está configurada")
        
        credentials_dict = json.loads(raw_json)

        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=["https://www.googleapis.com/auth/drive"]
        )

        service = build("drive", "v3", credentials=credentials)

        # 2. NUEVO ID de tu carpeta (extraído de tu link)
        FOLDER_ID = "15B9UQLcfqj1-x36ITnoLJUoaRuyxagAd"

        # Leer el contenido del archivo
        file_bytes = await file.read()
        file_stream = io.BytesIO(file_bytes)

        file_metadata = {
            "name": file.filename,
            "parents": [FOLDER_ID]
        }

        # MediaIoBaseUpload maneja la subida del stream de bytes
        media = MediaIoBaseUpload(file_stream, mimetype=file.content_type, resumable=True)

        uploaded_file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id",
            supportsAllDrives=True
        ).execute()

        file_id = uploaded_file.get("id")
        print(f"Éxito! Archivo subido con ID: {file_id}")

        return {"fileId": file_id, "status": "success"}

    except Exception as e:
        print("ERROR CRÍTICO:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


