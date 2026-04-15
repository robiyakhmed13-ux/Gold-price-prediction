# ❤️ Heart Disease Prediction

A machine learning project that predicts whether a patient **has heart disease or not** using **Logistic Regression**, based on clinical measurements.

---

## 📌 Project Overview

Heart disease is the leading cause of death worldwide. Early prediction using patient data can significantly improve outcomes. This binary classification project uses Logistic Regression — well-suited for medical binary classification tasks — to predict the presence of heart disease.

| Item | Detail |
|------|--------|
| **Algorithm** | Logistic Regression |
| **Task** | Binary Classification |
| **Dataset** | [Heart Disease Dataset – Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| **Target** | `target` — Healthy (0) / Has Disease (1) |

---

## 📂 Project Structure

```
heart_disease_prediction/
│
├── heart_disease_prediction.ipynb   # Jupyter Notebook (full walkthrough)
├── heart_disease_prediction.py      # Clean Python script
├── requirements.txt                 # Dependencies
├── heart_disease_data.csv           # Dataset (download from Kaggle)
├── eda_plots.png                    # EDA visualizations
├── correlation_heatmap.png          # Feature correlation heatmap
├── confusion_matrix.png             # Confusion matrix
└── README.md
```

---

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| `age` | Age of the patient |
| `sex` | Sex (Male=1, Female=0) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1=True, 0=False) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1=Yes, 0=No) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia type (0=Normal, 1=Fixed Defect, 2=Reversible Defect) |
| `target` | ✅ **Target** — 0=Healthy, 1=Has Disease |

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `heart_disease_data.csv` from Kaggle and place it in the project root.

### 4. Run
```bash
python heart_disease_prediction.py
```

---

## 🔄 Pipeline

```
Raw CSV Data
    │
    ▼
EDA — Target distribution, age/heart rate vs disease, heatmap
    │
    ▼
Feature / Target Split
    │
    ▼
Train / Test Split (90% / 10%, stratified)
    │
    ▼
Logistic Regression Training
    │
    ▼
Accuracy + Classification Report + Confusion Matrix
    │
    ▼
Single-patient Heart Disease Prediction
```

---

## 📈 Results

| Split | Accuracy |
|-------|----------|
| Training | ~85% |
| Test | ~82% |

---

## 🔮 Sample Prediction

```python
# (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
sample = (48, 0, 2, 130, 275, 0, 1, 139, 0, 0.2, 2, 0, 2)
result = predict_heart_disease(model, sample)
# Output: 💚 Prediction: Healthy Heart — No disease detected.
```

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas / numpy** — data processing
- **scikit-learn** — Logistic Regression, metrics, train/test split
- **seaborn / matplotlib** — visualization

---

## 🚀 Future Improvements

- [ ] Try SVM, Random Forest, or XGBoost for comparison
- [ ] Add feature scaling (`StandardScaler`) for potentially improved Logistic Regression performance
- [ ] Cross-validation (k-fold) for more robust evaluation
- [ ] Build a patient-facing Streamlit web app
- [ ] SHAP values for feature importance and model explainability

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and is not intended for medical diagnosis. Always consult a qualified medical professional for health concerns.

---

## 📄 License

MIT License

---

## 🙋 Author

**[Your Name]**  
[GitHub](https://github.com/your-username) | [LinkedIn](https://linkedin.com/in/your-profile)
