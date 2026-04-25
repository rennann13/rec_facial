from fastapi import FastAPI, UploadFile, File, Form
import face_recognition
import numpy as np
import json

app = FastAPI()

@app.get("/")
def main():
    return {"status": "Funcionando"}

@app.post("/gerar-vetor")
async def gerador_vetor(foto: UploadFile = File(...)):
    img = face_recognition.load_image_file(foto.file)

    econdings = face_recognition.face_encodings(img)

    if not econdings:
        return {
            "erro": "Nenhum rosto encontrado"
        }

    vetor = econdings[0].tolist()

    return {"vetor": vetor}

@app.post("/verify_face")
async def verify_face(
        vector_db: str = Form(...),
        foto: UploadFile = File(...)
):
    vector_db_list = json.loads(vector_db)
    vector_db_array = np.array(vector_db_list)

    image = face_recognition.load_image_file(foto.file)

    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return {"match": False, "error": "Nenhuma face encontrada na imagem."}

    vector_image = encodings[0]

    match = face_recognition.compare_faces([vector_db_array], vector_image, tolerance=0.6)[0]

    return {
        "match": bool(match),
    }