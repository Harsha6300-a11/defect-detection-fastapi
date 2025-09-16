# defect-detection-fastapi
FastAPI + TensorFlow defect detection API. Dockerized and deployed on Kubernetes.


# Defect Detection in Manufacturing using FastAPI, TensorFlow, Docker & Kubernetes

## 📌 Overview
This project implements a **computer vision system** to detect six types of surface defects 
(crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches) from the **NEU Surface Defect Dataset**.  
It combines **TensorFlow CNNs** for defect classification with a **FastAPI REST API** for serving predictions, 
containerized with **Docker**, and deployed on **Kubernetes** for scalable inference.

---

## 🚀 Features
- Trained TensorFlow CNN on **1,800+ images** (6 defect classes) with ~87% validation accuracy
- REST API endpoints for:
  - Single image prediction
  - Batch image prediction
  - Validation accuracy check
- Dockerized for portability
- Kubernetes deployment with replicas and load balancing
- Scalable inference (tested with 5+ replicas) and ~40% lower latency compared to single-container deployment

---

## 📂 Project Structure
defect-detection-fastapi/
│── app.py # FastAPI app (single + batch prediction endpoints)
│── train_tf.py # Model training script
│── defect_model.h5 # Saved trained model (not pushed to repo)
│── metrics.txt # Validation accuracy log
│── requirements.txt # Python dependencies
│── Dockerfile # Docker container definition
│── deployment.yaml # Kubernetes Deployment + Service
│── README.md # Project documentation
└── .gitignore # Ignore datasets, cache, and large files


---

## ⚙️ Setup Instructions

### 1️⃣ Create Environment
```bash
conda create -n defect_detection python=3.10
conda activate defect_detection
pip install -r requirements.txt

2️⃣ Dataset

Dataset used: NEU Surface Defect Database

Place dataset under:

dataset/NEU-DET/train/images/<class_folders>
dataset/NEU-DET/validation/images/<class_folders>

3️⃣ Train the Model
python train_tf.py


Model is saved as defect_model.h5.

🌐 Run FastAPI Locally
uvicorn app:app --reload


Docs: http://127.0.0.1:8000/docs

Health check: http://127.0.0.1:8000/

🐳 Run with Docker
docker build -t defect-detection .
docker run -p 8000:8000 defect-detection


Or pull from Docker Hub:

docker pull chintaguntavv/defect-detection:latest
docker run -p 8000:8000 chintaguntavv/defect-detection:latest

☸️ Deploy on Kubernetes

Apply deployment:

kubectl apply -f deployment.yaml


Check pods:

kubectl get pods


Access service:

minikube service defect-detection-service

📊 Results

  Dataset: 1,800+ images

  Classes: 6 defect types

  Validation accuracy: ~87%

  Scalable deployment: 5 replicas

  Latency improvement: 40% faster under concurrent requests

🛠️ Tech Stack

Languages: Python

Libraries: TensorFlow, Keras, NumPy, Pandas, OpenCV, FastAPI

Deployment: Docker, Kubernetes, Uvicorn

Tools: Conda, Git, Minikube

📌 Future Work

Add Grad-CAM heatmaps for model interpretability

Extend dataset with real-world industrial defect images

Integrate CI/CD for automated builds and deployments

