from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import io
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image as imgo
import tensorflow as tf
from tensorflow.keras.models import load_model
# tf environment variable setting
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


app = FastAPI()


# add the middleware to the app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)
model = load_model("model.h5")


class Image(BaseModel):
    image: str


@app.post("/show_image/")
def show_image(image: Image):
    return {"image": image.image}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict/")
def predict(image: Image):
    encoded_image = image.image
    encoded_image = encoded_image.split(",")[1]
    # print("Encoded image:", encoded_image)
    img = imgo.open(io.BytesIO(
        base64.decodebytes(bytes(encoded_image, "utf-8"))))
    img_arr = np.array(img)
    img_grayscale = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    img_to_feed = np.reshape(img_grayscale, (1, 28, 28, 1))
    print("Image to feed shape:", img_to_feed.shape)
    print("-"*100)
    prediction = model.predict(img_to_feed)
    predicted_label = np.argmax(prediction)

    print("Prediction:", prediction)
    print("Predicted label:", predicted_label)
    return {"prediction": "predicted_label"}
