import uvicorn
from fastapi import FastAPI, UploadFile, File
import cv2
import io
import tensorflow as tf
from PIL import Image
import numpy as np

app = FastAPI()

model = tf.keras.models.load_model("model2.h5")
classes = {0: "Healthy", 1: "Powdery", 2: "Rust", 3: "Unrecognized"}
@app.post("/predict")
async def predict(use_camera: bool = False, file: UploadFile = File(None)):
    if use_camera:
        # Capture image using the webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        # Convert image to PIL format
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    elif file is not None:
        # Read and preprocess the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    else:
        return {"error": "No image provided"}
    # Resize and preprocess the image
    image = image.resize((225, 225))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    # Make a prediction
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions)
    predicted_label = classes[predicted_index]
    return {"prediction": predicted_label}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)