from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

app = FastAPI()
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html") as f:
        return f.read()
model = ResNet50(weights="imagenet")

def generate_caption(img):
    img = img.resize((224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)

    predictions = decode_predictions(preds, top=1)[0]
    label = predictions[0][1]

    caption = f"A photo of a {label}"
    return caption


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    img = Image.open(file.file)

    caption = generate_caption(img)

    return {"caption": caption}