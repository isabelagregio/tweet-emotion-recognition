# Emotion Detection API 🎯

A FastAPI web application for detecting emotions in text, powered by a **Bidirectional LSTM** model trained with TensorFlow/Keras.

This project was built to complement the learnings and guidance obtained in the "Tweet Emotion Recognition with TensorFlow" course on Coursera, which has the original version of the notebook used.

Link to course: https://www.coursera.org/projects/tweet-emotion-tensorflow 
## 📌 Features

-   Classifies text into: `fear`, `anger`, `love`, `sadness`, `surprise`, `joy`
-   Simple HTML form interface (Jinja2)
-   Pre-trained model & tokenizer included
-  Containerization with Docker

## 🛠 Tech Stack

-   **Backend:** FastAPI
-   **Model:** TensorFlow/Keras (Bi-LSTM)
-   **Frontend:** Jinja2 Templates
-   **Data Handling:** Pandas, NumPy
-   **Architecture:** Embedding + Bidirectional LSTM (x2) + Dense Softmax

## 🚀 Running the Application

1️⃣ **Build the image**:

```bash
docker build -t emotion-api .
```

 

2️⃣ **Run the container**:

```bash
docker run -d -p 8000:8000 emotion-api
```
  

3️⃣ **Access in browser**:

```bash
http://localhost:8000
```

