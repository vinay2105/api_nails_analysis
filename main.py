from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
import numpy as np
from PIL import Image
import requests
import io

model = load_model('my_trained_model.keras')

class_names = ['acrall lentiginous melanoma', 'blue finger', 'onychogryphosis', 'healthy nail', 'clubbing', 'pitting']

app = FastAPI()

class ImageURL(BaseModel):
    url: str

@app.post("/predict")
async def predict_nail_condition(image_url: ImageURL):
    try:
        response = requests.get(image_url.url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Image could not be retrieved from the URL")
        
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values
        
    
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = class_names[class_idx]
        
        return {"predicted_class": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

