<h1 align="center">⚙️ MaintelliSense</h1>
<p align="center">
  <i>Predictive Maintenance Intelligence – Prevent Failures Before They Happen</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Enabled-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3.9%2B-yellow?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge">
</p>

---

## 📖 Overview  
**MaintelliSense** is an intelligent predictive maintenance system that leverages **machine learning** to forecast potential equipment failures before they occur.  
It enables industries to reduce downtime, optimize maintenance schedules, and save operational costs by making **data-driven decisions**.

---

## ✨ Core Highlights  
- 🔹 **Realistic Data Simulation** – Generate industrial-grade sensor data (temperature, vibration, etc.)  
- 🔹 **Automated Data Preprocessing** – Cleaning, transformation, and feature selection  
- 🔹 **SMOTE for Imbalance Handling** – Ensure fair model learning  
- 🔹 **Powerful Prediction Engine** – Built with Random Forest & customizable ML models  
- 🔹 **Visual Analytics Dashboard** – Real-time machine status & failure probability  
- 🔹 **Automated PPT Reports** – Instantly export analysis results for stakeholders  

---

## 🧠 How It Works  
1. **Data Simulation** – Create synthetic sensor datasets  
2. **Preprocessing** – Handle missing values, normalize features  
3. **Class Balancing** – Apply SMOTE to fix class imbalance  
4. **Model Training** – Train a Random Forest model  
5. **Evaluation** – Generate accuracy, recall, and precision metrics  
6. **Reporting & Dashboard** – Export PPT & view live metrics in Streamlit  

---

## 🛠 Tech Stack  
| Component | Technology |
|-----------|------------|
| Language  | Python 3.9+ |
| Data Handling | Pandas, NumPy |
| Machine Learning | scikit-learn, imbalanced-learn |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Reporting | python-pptx |

---

## 📂 Project Structure  
```bash
MaintelliSense/
├── data/            # Raw & processed datasets
├── models/          # Trained ML models
├── reports/         # Evaluation reports
├── presentation/    # Auto-generated PPT files
├── src/             # Source code scripts
└── main.py          # Main execution script
