# 🛡️ Online Fraud Detection System (Streamlit App)

This project provides an interactive Streamlit web application to predict **online financial frauds** using Machine Learning (ML), Deep Learning (DL), and Reinforcement Learning (RL). It enables real-time prediction and batch evaluation based on engineered transaction features.

---

## 🚀 Features

- 📁 Upload reference data (`onlinefraud.csv`) or load from working directory
- 🧾 Fill in form for individual transaction simulation
- 🧪 Automatic **feature engineering** from raw fields
- 📊 Correlation matrix + custom feature selection
- 🧠 Prediction via:
  - ✅ Uploading a trained model (`.pkl`, `.h5`, `.zip`, `.npy`)
  - ✅ Training a new model: ML / DL / Hyperparameter / RL
- 📦 Batch fraud prediction using uploaded `.xlsx`
- 📉 Output predictions with confidence levels
- 📥 Export results as `.xlsx`

---

## 📂 Directory Structure

```
├── app.py                   # Main Streamlit app
├── onlinefraud.csv          # Sample raw dataset (optional, for reference)
├── requirements.txt         # Full dependency list
├── models/                  # Folder to store trained models (optional)
└── README.md
```

---

## 🛠️ Setup Instructions

### 1. Clone and Setup Environment

```bash
git clone https://github.com/your-username/online-fraud-ml.git
cd online-fraud-ml
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

---

## 🧾 Sample Transaction Prediction (Manual Entry)

Users input:

```python
step = 5
amount = 95000
oldbalanceOrg = 95000
newbalanceOrg = 0
...
```

The app computes features like:

```python
balance_diff_org = oldbalanceOrg - newbalanceOrg
is_large_txn = int(amount > 50000)
...
```

And runs prediction using selected model:

```python
model = joblib.load("fraud_ml_ensemble.pkl")
X_scaled = scaler.transform(input_data)
pred = model.predict(X_scaled)
```

Result:

```
Prediction: FRAUD | Confidence: 97.8%
```

---

## 🖼️ Demo Screenshots / GIF

### 🔍 App Preview

![App Screenshot](https://raw.githubusercontent.com/your-username/online-fraud-ml/main/demo_screenshot.png)

### 🎞️ End-to-End Demo

![Demo GIF](https://raw.githubusercontent.com/your-username/online-fraud-ml/main/demo.gif)

---

## 📦 Batch Fraud Prediction

- Upload `.xlsx` with raw transaction data
- App performs full feature engineering automatically
- Choose model and features to predict
- View a random sample of 50 results
- Download full predictions:

```text
batch_predictions.xlsx
```

---

## 🧠 Supported Algorithms

| Type | Options                                  |
| ---- | ---------------------------------------- |
| ML   | RandomForest, Ensemble (RF + XGB + LGBM) |
| DL   | MLP (Keras), MLP + KerasTuner (Grid)     |
| RL   | Q-Learning, DQN via Stable-Baselines3    |

---

## 📌 Notes

- 🛠️ Trained models can be uploaded or saved after training
- 🔁 Feature engineering follows the same logic across all predictions
- 📉 Confidence for Q-Learning estimated using sigmoid of score

---

## 📦 Requirements

See `requirements.txt` or install key packages:

```txt
streamlit
pandas
numpy==1.23.5
scikit-learn
xgboost
lightgbm
tensorflow==2.11.0
keras==2.11.0
stable-baselines3
gymnasium
openpyxl
```

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Please ensure any new models or logic follow the app’s design.

---

## 📜 License

[MIT License](LICENSE)

---

## 👨‍💻 Author

Built by [@AmbOlatunji](https://github.com/ambolatunji) with ❤️ and coffee ☕
