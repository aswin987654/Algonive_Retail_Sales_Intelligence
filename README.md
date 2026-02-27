# 📊 Retail Sales Intelligence Platform

## 🚀 Overview

Retail Sales Intelligence is a production-ready machine learning system designed to forecast weekly revenue at:

- Store level  
- Store + Category level  

The platform helps retail businesses:

- Improve inventory planning  
- Optimize marketing strategies  
- Reduce stockouts and overstock  
- Compare ML performance against baseline forecasting  

The system includes:

- Forecasting models  
- Executive dashboard  
- Baseline comparison  
- Downloadable PDF executive reports  
- Dockerized deployment  

---

## 🎯 Business Problem

Retail revenue fluctuates due to:

- Seasonality  
- Customer demand variation  
- Category-level shifts  
- Store-specific trends  

Naive forecasting methods (like last week’s revenue) often fail to capture these patterns.

This system compares:

- A baseline naive forecasting model  
- A Random Forest ML model  

And quantifies improvement using:

- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- SMAPE (Symmetric Mean Absolute Percentage Error)  

---

## 🧠 Solution Architecture

### Data Pipeline

1. Data preprocessing (aggregation to weekly level)
2. Feature engineering:
   - Lag features (4-week, 12-week)
   - Rolling means
   - Time-based features (year, week number)
3. Time-aware train/test split
4. Model training using Random Forest Regressor
5. Baseline comparison
6. Dashboard visualization
7. Executive PDF generation

---

## 📈 Features

✔ Weekly revenue forecasting  
✔ Store-level filtering  
✔ Category-level forecasting  
✔ Baseline vs ML comparison  
✔ Executive metrics display  
✔ Downloadable PDF report  
✔ Fully Dockerized  
✔ Versioned releases  

---

## 🏗 Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- Streamlit  
- Matplotlib  
- Docker
- Docker Hub 

---

## 📦 How to Run (Docker)

Pull the production release:

```bash
docker pull aswinr7191/retail-forecast-app:v2.1.0
docker run -p 8501:8501 aswinr7191/retail-forecast-app:v2.1.0
