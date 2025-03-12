from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO
from fastapi.responses import Response
from io import BytesIO
from PIL import Image

app = FastAPI()

# Carregar modelo YOLO pré-treinado
model = YOLO("yolov8n.pt")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Ler a imagem
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Fazer a detecção com YOLO
    results = model(image)[0]
    
    # Desenhar as bounding boxes e rótulos
    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        label = f"{result.cls[0].item():.0f}"  # Classe detectada
        conf = f"{result.conf[0].item():.2f}"  # Confiança da detecção
        text = f"{label} ({conf})"
        
        # Desenhar retângulo
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Converter imagem processada para bytes
    _, img_encoded = cv2.imencode(".jpg", image)
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
