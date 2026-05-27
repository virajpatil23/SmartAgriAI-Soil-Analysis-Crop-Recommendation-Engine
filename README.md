# 🌾 SmartAgriAI — Soil Health Analysis & Crop Recommendation Engine

> End-to-end Deep Learning system for soil fertility prediction, crop recommendation,
> and fertilizer scheduling. Built with TensorFlow/Keras, Flask, and Scikit-learn.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![Accuracy](https://img.shields.io/badge/DNN_Accuracy-99.39%25-brightgreen)

---

## 🔍 Overview

SmartAgriAI is a machine learning system that analyzes soil health parameters
and provides actionable recommendations for farmers. It predicts soil fertility,
recommends suitable crops, and generates fertilizer schedules based on 23 input
parameters including NPK levels, pH, micronutrients, and climate data.

---

## ✨ Features

- **Soil Fertility Prediction** — Binary classification (fertile/infertile),
  regression-based fertility rate, and 3-class fertility classification
- **Crop Recommendation** — Suggests top 5 crops based on soil and climate conditions
- **Fertilizer Scheduling** — Season-specific NPK and micronutrient application plans
- **Web Interface** — Full-stack Flask app with responsive HTML/CSS frontend
- **Multi-Dataset Pipeline** — Merges and preprocesses 3 agricultural datasets
  with KNN imputation and label encoding

---

## 🧠 Model Architecture

Multi-task Deep Neural Network (DNN) with shared layers and 3 task-specific branches:
Input (23 features)
↓
Dense(256, ReLU) → BatchNorm → Dropout(0.3)
Dense(128, ReLU) → BatchNorm → Dropout(0.2)
Dense(64,  ReLU) → BatchNorm → Dropout(0.2)
↓              ↓                ↓
is_fertile     fertility_rate   fertility_class
(sigmoid)      (sigmoid)        (softmax, 3)

### Model Performance

| Model | Accuracy |
|---|---|
| **Custom DNN (ours)** | **99.39%** |
| Random Forest | 96.8% |
| SVM | 94.2% |

---

## 📊 Dataset

- **Sources:** 3 agricultural datasets merged via outer join on common columns
- **Samples:** 5,000+ soil records
- **Features:** 23 parameters — N, P, K, pH, EC, OC, OM, Zn, Fe, Cu, Mn,
  Sand, Silt, Clay, CaCO3, CEC, S, B, Temperature, Humidity, Rainfall,
  Soil Type, Season
- **Preprocessing:** KNN imputation (k=5), Label encoding, StandardScaler

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML/DL | TensorFlow 2.x, Keras, Scikit-learn |
| Data Processing | Pandas, NumPy, KNNImputer |
| Backend | Flask, Python 3.10 |
| Frontend | HTML5, CSS3, Font Awesome |
| Model Persistence | .h5 (Keras), Pickle (metadata) |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python Train_Fertility_model_9.py
```

### Run the Web App
```bash
python app.py
# Open http://localhost:5000
```

### Make a Prediction (CLI)
```bash
python Predict_10.py
```
---

## 📈 Results

The system provides:
- **Fertility score** (0–1 continuous) with binary classification
- **Top 5 crop recommendations** based on soil and climate conditions
- **Complete fertilizer plan** — primary (NPK), secondary, and organic options
- **Season-specific application schedule** — Kharif, Rabi, and Zaid

---

## 🌱 Sample Input → Output

**Input:** pH 6.8, N=120 mg/kg, P=45 mg/kg, K=180 mg/kg,
Temperature 24°C, Rainfall 850mm, Loamy soil, Kharif season

**Output:**
- Fertility: High (0.89)
- Recommended crops: Rice, Wheat, Maize, Soybean, Groundnut
- Fertilizer: Urea 100 kg/ha (basal) + DAP 50 kg/ha + MOP 75 kg/ha
- Schedule: Basal → 30 DAS → 60 DAS

---

## 🔮 Future Work

- [ ] Fine-tune with BioBERT-style domain-specific embeddings
- [ ] Add MLflow experiment tracking
- [ ] Deploy on cloud (AWS/Render)
- [ ] Mobile-responsive UI upgrade
- [ ] Add real-time weather API integration

---

## 👤 Author

**Viraj Raosaheb Patil**
B.Tech CSE (AI) | Nutan College of Engineering and Research, Pune

[![GitHub](https://img.shields.io/badge/GitHub-virajpatil23-black)](https://github.com/virajpatil23)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-virajpatil-blue)](https://linkedin.com/in/virajpatil)

---

## 📄 License

MIT License — free to use and modify with attribution.
