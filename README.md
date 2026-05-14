# 🍔 Food Delivery Analytics & Intelligent Recommendation System

An AI-powered analytics platform built with Streamlit for food delivery business intelligence.

---

## 🚀 Quick Start

```bash
cd FoodDeliveryAI
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
FoodDeliveryAI/
├── app.py                  ← Main entry + model training gateway
├── requirements.txt
├── .streamlit/config.toml  ← Dark theme config
├── dataset/
│   └── Final_New_dataset.csv
├── models/                 ← Auto-generated on first run
├── pages/
│   ├── 1_Dashboard.py      ← Executive KPI dashboard
│   ├── 2_Data_Analytics.py ← EDA + distributions
│   ├── 3_Association_Rules.py ← Apriori/FP-Growth + recommender
│   ├── 4_Delivery_Prediction.py ← ML classification
│   ├── 5_Customer_Segmentation.py ← KMeans clustering
│   ├── 6_ANN_Prediction.py ← Deep learning time prediction
│   ├── 7_Business_Insights.py ← Auto-generated insights
│   ├── 8_Recommendation_System.py ← Swiggy-style engine
│   └── 9_About_Project.py  ← Project overview & team
└── utils/
    ├── data_loader.py      ← Cached CSV loading
    ├── preprocessing.py    ← Feature engineering
    ├── model_trainer.py    ← Train & persist all models
    ├── visualization.py    ← Plotly chart factory
    ├── recommendation.py   ← Rule-based recommender
    ├── prediction.py       ← Model inference wrappers
    ├── clustering.py       ← PCA & cluster profiling
    ├── insights.py         ← NL insight generator
    └── helper.py           ← CSS, KPI cards, downloads
```

---

## 🤖 First Run — Model Training

On first launch, navigate to the home page and click **"🚀 Train Models"**.

This will train (once, ~60-120 seconds):
- 4 Classification models (LR, DT, Random Forest, SVM)
- KMeans + Hierarchical clustering
- ANN for delivery time regression
- Apriori/FP-Growth association rules

Models are saved to `models/` and auto-loaded on every subsequent launch.

---

## 📊 Pages Overview

| Page | Description |
|---|---|
| 🏠 Dashboard | KPI cards, revenue trends, delivery stats |
| 📊 Data Analytics | EDA, missing values, distributions, outliers |
| 🔗 Association Rules | Market basket, food recommender widget |
| 🎯 Delivery Prediction | Real-time ML prediction + model comparison |
| 👥 Customer Segmentation | Cluster profiles + customer lookup |
| 🧠 ANN Prediction | Delivery time estimation + training curves |
| 💡 Business Insights | Auto-generated executive insights |
| 🛍️ Recommendation System | Restaurant + food item recommendations |
| ℹ️ About Project | Architecture, team, workflow |

---

## 🛠️ Tech Stack

- **UI**: Streamlit + Custom CSS (dark theme, Inter font)
- **ML**: Scikit-learn (RF, LR, DT, SVM, KMeans)
- **Deep Learning**: TensorFlow/Keras (ANN)
- **Association Mining**: MLxtend (FP-Growth, Apriori)
- **Visualization**: Plotly Express + Graph Objects
- **Data**: Pandas, NumPy
- **Persistence**: Joblib, Keras save

---

## 📦 Requirements

Python 3.9+, see `requirements.txt` for full list.

---

*Built as a final year college project — Food Delivery Analytics & Intelligent Recommendation System (2024)*
