from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import os
import numpy as np
import pickle

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = tf.keras.models.load_model(os.path.join("model", "tf_model.keras"))

with open(os.path.join("model", "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

labels = ['fear', 'anger', 'love', 'sadness', 'surprise', 'joy']

MAX_LEN = 50

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    return padded

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
def form_post(request: Request, text: str = Form(...)):
    X = preprocess_text(text)
    prediction = model.predict(X)
    predicted_index = int(np.argmax(prediction, axis=1)[0])
    predicted_label = labels[predicted_index]
    return templates.TemplateResponse("form.html", {"request": request, "result": predicted_label, "text": text})
