# Soil-Health-Monitoring-system

# 🌾 Soil Health Analysis & Crop Recommendation using Deep Learning (DNN)

This project uses a Deep Neural Network (DNN) to analyze soil health, predict soil fertility, and provide tailored crop and organic fertilizer recommendations. Designed to help farmers improve crop productivity through scientific soil analysis, this system also suggests crop rotation plans for sustainable farming.

---

## 📌 Objective

- Predict soil **fertility index** based on nutrient and environmental data.
- Recommend **suitable crops** (legumes, cereals, vegetables).
- Suggest **organic fertilizers** based on deficiencies.
- Generate **crop rotation plans** to preserve soil health.

---

## 🧠 Model: Deep Neural Network (DNN)

We use a regression-based DNN model built with TensorFlow/Keras. The model is trained on real-world soil data and predicts a continuous **fertility index**.

### ✅ Features Used:
- **Nutrient Parameters**: `N`, `P`, `K`, `Zn`, `Fe`, `Cu`, `Mn`, `B`, `S`, `OC`, `EC`
- **Environmental Factors**: `temperature`, `humidity`, `ph`, `rainfall`

### 🎯 Output:
- Fertility score (0 to 1)
- Top 3 crop recommendations
- Organic fertilizer suggestions
- Complete year crop rotation plan


