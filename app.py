from fastapi import FastAPI, UploadFile, File, Form
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

        credentials_dict = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])

        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=["https://www.googleapis.com/auth/drive"]
        )

        service = build("drive", "v3", credentials=credentials)

        file_bytes = await file.read()
        file_stream = io.BytesIO(file_bytes)

        file_metadata = {
            "name": file.filename,
            "parents": ["15B9UQLcfqj1-x36ITnoLJUoaRuyxagAd"]
        }

        media = MediaIoBaseUpload(file_stream, mimetype=file.content_type)

        uploaded_file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id"
        ).execute()

        file_id = uploaded_file.get("id")

        return {"fileId": file_id}

    except Exception as e:
        return {"error": str(e)}
