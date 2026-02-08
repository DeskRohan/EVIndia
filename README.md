# ⚡ EVIndia – EV Adoption & Charging Station Optimization

EVIndia is a data analytics and AI-based dashboard that analyzes Electric Vehicle (EV) adoption trends in India and identifies optimal locations for EV charging stations using machine learning (K-Means clustering).

---

## 📌 Project Overview

With the rapid growth of Electric Vehicles in India, efficient placement of charging stations is critical. This project analyzes EV adoption data and charging infrastructure to support data-driven EV infrastructure planning.

The project focuses on:
- Understanding EV sales growth trends
- Visualizing charging station distribution
- Optimizing new charging hub locations using AI
- Forecasting future EV adoption

---

## 📊 Datasets Used

1. **EV Sales Data**  
   State-wise and vehicle-category-wise EV sales data used for trend analysis and forecasting.

2. **Charging Station Data**  
   A synthetic but realistic dataset (~29,000 records) representing projected EV charging infrastructure distribution across India (late 2025).

> ⚠️ Note: Charging station data is synthetic and used for academic and analytical purposes.

---

## 🎯 Objectives

- Analyze EV adoption growth over time and by state  
- Visualize the distribution of existing charging infrastructure  
- Identify optimal zones for new charging stations using clustering  
- Predict future EV adoption trends  

---

## 🧠 Machine Learning Techniques

- **K-Means Clustering**  
  Used for charging station optimization by grouping nearby stations and identifying central hub locations.

- **Linear Regression**  
  Used for forecasting future EV adoption trends.  
  Model performance is evaluated using the R² score.

---

## 🖥️ Dashboard Features

- 📈 EV adoption trends (time-series analysis)
- 🗺️ Interactive Leaflet maps with marker clustering
- 🎯 AI-based charging hub optimization using a slider
- 🔮 Future EV sales forecast
- 📊 Clean, minimal, light-mode Streamlit UI

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Pandas, NumPy  
- Scikit-learn  
- Plotly  
- Folium (Leaflet Maps)

---

## ▶️ How to Run Locally

1. Install dependencies:
pip install -r requirements.txt

2. Run the application:
streamlit run app.py

---

## 📈 Model Evaluation

- Linear Regression model achieved an R² score of approximately **0.80**
- Indicates good trend capture for planning-level forecasting

---

## 🚀 Future Enhancements

- Automatic selection of optimal number of clusters
- EV-to-charging-station demand ratio analysis
- CO₂ emission reduction estimation
- Integration with real-time datasets

---

## 👤 Author

**Rohan**  
Computer Science Engineering (CSE)

---

## 📄 License

This project is intended for academic and learning purposes.
