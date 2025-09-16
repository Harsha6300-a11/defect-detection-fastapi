# defect-detection-fastapi
FastAPI + TensorFlow defect detection API. Dockerized and deployed on Kubernetes.


# Defect Detection in Manufacturing using FastAPI, TensorFlow, Docker & Kubernetes

## 📌 Overview
Surface defects in manufacturing, such as scratches, inclusions, or rolled-in scales, can significantly reduce product quality and reliability. Manual inspection is often slow, inconsistent, and costly.  
This project provides an **AI-powered defect detection system** that leverages **Convolutional Neural Networks (CNNs)** with **TensorFlow** to automatically classify defects in steel surfaces. The solution is exposed via a **FastAPI REST API**, packaged with **Docker**, and deployed on **Kubernetes** for scalability and resilience.

## 🎯 Motivation
Traditional defect inspection methods are manual and error-prone, making it difficult to ensure high standards in mass production.  
By automating defect detection, manufacturers can:
- Improve product quality and reduce recalls  
- Save time and operational costs  
- Scale inspection to large production lines  
- Enable real-time defect monitoring  


## 🚀 Features
- Trained TensorFlow CNN on **1,800+ images** (6 defect classes) with ~87% validation accuracy
- REST API endpoints for:
  - Single image prediction
  - Batch image prediction
  - Validation accuracy check
- Dockerized for portability
- Kubernetes deployment with replicas and load balancing
- Scalable inference (tested with 5+ replicas) and ~40% lower latency compared to single-container deployment



## 📂 Project Structure
```
defect-detection-fastapi/
│── app.py # FastAPI app (single + batch prediction endpoints)
│── train_tf.py # Model training script
│── defect_model.h5 # Saved trained model (not pushed to repo)
│── download_dataset.py 
│── metrics.txt # Validation accuracy log
│── requirements.txt # Python dependencies
│── Dockerfile # Docker container definition
│── deployment.yaml # Kubernetes Deployment + Service
│── README.md # Project documentation
└── .gitignore # Ignore datasets, cache, and large files

```


## ⚙️ Setup Instructions

**1️⃣ Create Environment**
 ```bash
conda create -n defect_detection python=3.10  
conda activate defect_detection  
pip install -r requirements.txt  
```

**2️⃣ Dataset**

Dataset used: NEU Surface Defect Database  

Place dataset under:  
 ```bash
dataset/NEU-DET/train/images/<class_folders>  
dataset/NEU-DET/validation/images/<class_folders>  
```
    After downloading the dataset , it might not classsify images as 6 classes , in that case use download_dataset.py or else ignore it 
    
**3️⃣ Train the Model**

Run this file 
```bash
                           python train_tf.py
```

-->Model is saved as defect_model.h5.

🌐 Run FastAPI Locally
 ```bash
  uvicorn app:app --reload
 ```
Docs: http://127.0.0.1:8000/docs
      
Health check: http://127.0.0.1:8000/


🐳 **Run with Docker**

```bash
docker build -t defect-detection .
docker run -p 8000:8000 defect-detection
```
  
  Or pull from Docker Hub:
  ```bash
docker pull chintaguntavv/defect-detection:latest
docker run -p 8000:8000 chintaguntavv/defect-detection:latest
```

**☸️ Deploy on Kubernetes**

  **Apply deployment**:
  ```bash
  kubectl apply -f deployment.yaml
  ```
 
  **Check pods**:
   ```bash
  kubectl get pods
  ```
  
  **Access service**:
   ```bash
  minikube service defect-detection-service
  ```

**📊 Results**

  **Dataset:** 1,800+ images  
  
  **Classes:** 6 defect types  
  
  **Validation accuracy:** ~87%  
  
  **Scalable deployment:** 5 replicas  
  
  **Latency improvement:** 40% faster under concurrent requests  


**🛠️ Tech Stack**

  **Languages:** Python  
  
  **Libraries:** TensorFlow, Keras, NumPy, Pandas, OpenCV, FastAPI  
  
  **Deployment:** Docker, Kubernetes, Uvicorn  
  
  **Tools:** Conda, Git, Minikube  

## 🔍 Use Cases
- **Steel industry**: Automated inspection of rolled steel surfaces  
- **Electronics manufacturing**: Detecting defects in silicon wafers or PCB boards  
- **Automotive**: Surface quality assurance for body panels and parts  
- **General manufacturing**: Any high-throughput environment where surface quality matters  

## 📌 Future Work

- [ ] Add Grad-CAM heatmaps for model interpretability  
- [ ] Extend dataset with real-world industrial defect images  
- [ ] Integrate CI/CD for automated builds and deployments  
- [ ] Implement model monitoring & logging for production inference  
- [ ] Optimize model with TensorRT / ONNX for faster inference  
- [ ] Explore active learning to continuously improve the dataset  
- [ ] Add support for streaming data (real-time video defect detection)  
- [ ] Integrate with cloud platforms (AWS/GCP/Azure) for large-scale deployment  
 
