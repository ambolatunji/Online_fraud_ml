import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from stable_baselines3 import DQN

st.set_page_config(page_title="Fraud Predictor", layout="wide")
st.title("🔍 Online Fraud Prediction")

# --- SECTION 1: Upload CSV as reference ---
st.header("1. Upload Online Fraud CSV")
raw_file = st.file_uploader("onlinefraud.csv", type=['csv'])
if raw_file:
    df_ref = pd.read_csv(raw_file)
    st.dataframe(df_ref.head())

# --- SECTION 2: Manual Input Form ---
st.header("2. Input Transaction Details")
with st.form("txn_form"):
    col1, col2 = st.columns(2)
    with col1:
        step = st.number_input("Step", min_value=0)
        type_ = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "DEBIT", "PAYMENT", "CASH_IN"])
        amount = st.number_input("Amount", min_value=0.0)
        oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0)
        newbalanceOrg = st.number_input("New Balance (Origin)", min_value=0.0)
    with col2:
        nameOrig = st.text_input("Origin Name", value="C12345")
        nameDest = st.text_input("Destination Name", value="M67890")
        oldbalanceDest = st.number_input("Old Balance (Dest)", min_value=0.0)
        newbalanceDest = st.number_input("New Balance (Dest)", min_value=0.0)
        isFlaggedFraud = st.selectbox("Flagged Fraud", [0, 1])
    submitted = st.form_submit_button("Run Feature Engineering")

# --- SECTION 3: Feature Engineering ---
if submitted:
    st.header("3. Engineered Features & Correlation")

    def feature_engineering(row):
        row['balance_diff_org'] = row['oldbalanceOrg'] - row['newbalanceOrg']
        row['balance_diff_dest'] = row['newbalanceDest'] - row['oldbalanceDest']
        row['amount_diff_org'] = row['amount'] - row['balance_diff_org']
        row['amount_diff_dest'] = row['amount'] - row['balance_diff_dest']
        row['txn_ratio'] = row['amount'] / (row['oldbalanceOrg'] + 1e-6)
        row['is_sender_zero_bal'] = int(row['oldbalanceOrg'] == 0)
        row['is_receiver_zero_before'] = int(row['oldbalanceDest'] == 0)
        row['is_receiver_exact_amount'] = int(row['newbalanceDest'] - row['oldbalanceDest'] == row['amount'])
        row['is_large_txn'] = int(row['amount'] > 50000)
        row['org_to_dest_same'] = int(nameOrig[:1] == nameDest[:1])
        row['sender_is_customer'] = int(nameOrig.startswith("C"))
        row['receiver_is_customer'] = int(nameDest.startswith("C"))
        row['receiver_is_merchant'] = int(nameDest.startswith("M"))
        row['risk_combo'] = row['is_receiver_zero_before'] & row['is_large_txn'] & row['receiver_is_customer']
        row['is_night'] = int(step % 24 <= 6)
        return row

    input_dict = {
        'step': step, 'type': type_, 'amount': amount, 'nameOrig': nameOrig,
        'oldbalanceOrg': oldbalanceOrg, 'newbalanceOrg': newbalanceOrg,
        'nameDest': nameDest, 'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest, 'isFlaggedFraud': isFlaggedFraud
    }
    input_df = pd.DataFrame([input_dict])
    engineered = feature_engineering(input_df.copy())

    drop_cols = ['step', 'type', 'nameOrig', 'nameDest']
    feat_df = engineered.drop(columns=drop_cols)
    st.dataframe(feat_df)

    corr = feat_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    selected_features = st.multiselect("Select features for modeling", list(feat_df.columns), default=list(feat_df.columns))

    st.header("4. Choose Prediction Method")
    mode = st.radio("Choose Mode", ["Upload Model", "Train New Model"])

    X = feat_df[selected_features]

    if mode == "Train New Model":
        st.subheader("⚙️ Choose Model Type")
        model_type = st.selectbox("Model Type", ["ML", "DL", "ML (Hyperparameter)", "DL (Hyperparameter)", "RL (Q-Learning)", "RL (DQN)"])

        st.markdown("⚠️ Hyperparameter tuning may take longer to train. Please be patient.")

        if model_type in ["ML", "ML (Hyperparameter)"]:
            if model_type == "ML":
                ml_algo = st.selectbox("Select ML Algorithm", ["RandomForest", "Ensemble (RF + XGB + LGBM)"])
                run_btn = st.button("Train ML Model")
            else:
                ml_algo = "RandomForest"
                run_btn = st.button("Run ML Hyperparameter Tuning")

            if run_btn:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                y_train = [0, 1] * 10

                if model_type == "ML (Hyperparameter)":
                    param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 10]}
                    grid = GridSearchCV(RandomForestClassifier(), param_grid, scoring='roc_auc', cv=2)
                    grid.fit(X_scaled, y_train[:len(X_scaled)])
                    model = grid.best_estimator_
                else:
                    if ml_algo == "RandomForest":
                        model = RandomForestClassifier(class_weight='balanced').fit(X_scaled, y_train[:len(X_scaled)])
                    else:
                        rf = RandomForestClassifier()
                        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                        lgbm = LGBMClassifier()
                        model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)], voting='soft')
                        model.fit(X_scaled, y_train[:len(X_scaled)])

                proba = model.predict_proba(X_scaled)[0][1]
                label = model.predict(X_scaled)[0]
                st.success(f"Prediction: {'FRAUD' if label==1 else 'LEGIT'} | Confidence: {proba*100:.2f}%")
                if st.button("Save Model"):
                    joblib.dump(model, "trained_ml_model.pkl")
                    st.success("Model saved as trained_ml_model.pkl")

        elif model_type in ["DL", "DL (Hyperparameter)"]:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            y_train = np.array([0, 1] * 10)[:len(X)]

            if model_type == "DL":
                if st.button("Train Simple MLP"):
                    model = Sequential([
                        Dense(64, activation='relu', input_shape=(X.shape[1],)),
                        Dense(32, activation='relu'),
                        Dense(1, activation='sigmoid')
                    ])
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    model.fit(X_scaled, y_train, epochs=10, batch_size=16, verbose=0)
                    proba = model.predict(X_scaled)[0][0]
                    st.success(f"Prediction: {'FRAUD' if proba > 0.5 else 'LEGIT'} | Confidence: {proba*100:.2f}%")
                    if st.button("Save Model"):
                        model.save("trained_dl_model.h5")
                        st.success("Model saved as trained_dl_model.h5")

            else:
                st.warning("Grid search on units (32, 64, 128)")
                def create_dl_model(units=64):
                    model = Sequential()
                    model.add(Dense(units, activation='relu', input_shape=(X.shape[1],)))
                    model.add(Dense(1, activation='sigmoid'))
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    return model

                model = KerasClassifier(build_fn=create_dl_model, verbose=0)
                grid = GridSearchCV(model, param_grid={'units': [32, 64, 128]}, cv=2)
                grid.fit(X_scaled, y_train)
                best_model = grid.best_estimator_
                proba = best_model.model.predict(X_scaled)[0][0]
                st.success(f"[DL-HP] Prediction: {'FRAUD' if proba > 0.5 else 'LEGIT'} | Confidence: {proba*100:.2f}%")
                if st.button("Save Model"):
                    best_model.model.save("trained_dl_hp_model.h5")
                    st.success("Model saved as trained_dl_hp_model.h5")

        elif model_type == "RL (Q-Learning)":
            st.info("Training Q-Learning agent")
            weights = np.random.randn(X.shape[1])
            y_train = np.array([0, 1] * 10)[:len(X)]
            for epoch in range(100):
                for i in range(len(X)):
                    score = np.dot(weights, X.iloc[i])
                    pred = int(score > 0)
                    reward = 1 if pred == y_train[i] else -1
                    weights += 0.01 * reward * X.iloc[i]
            score = np.dot(weights, X.iloc[0])
            confidence = 1 / (1 + np.exp(-score))
            label = int(confidence > 0.5)
            st.success(f"Prediction (Q-Learning): {'FRAUD' if label==1 else 'LEGIT'} | Confidence: {confidence*100:.2f}%")
            if st.button("Save Q-Learning Model"):
                np.save("trained_qlearning_weights.npy", weights)
                st.success("Model saved as trained_qlearning_weights.npy")

        elif model_type == "RL (DQN)":
            st.warning("Train and save RL model via DQN externally due to environment complexity.")
            st.markdown("Use our RL notebook to train and export `dqn_fraud_model.zip`")

    elif mode == "Upload Model":
        model_file = st.file_uploader("Upload trained model (.pkl, .h5)", type=["pkl", "h5", "zip", "npy"])
        if model_file and st.button("Run Prediction"):
            if model_file.name.endswith(".pkl"):
                model = joblib.load(model_file)
                scaler = joblib.load("fraud_scaler.pkl")
                X_scaled = scaler.transform(X)
                proba = model.predict_proba(X_scaled)[0][1]
                label = model.predict(X_scaled)[0]
                st.success(f"Prediction: {'FRAUD' if label==1 else 'LEGIT'} | Confidence: {proba*100:.2f}%")

            elif model_file.name.endswith(".h5"):
                model = load_model(model_file)
                scaler = joblib.load("fraud_scaler.pkl")
                X_scaled = scaler.transform(X)
                proba = model.predict(X_scaled)[0][0]
                st.success(f"Prediction: {'FRAUD' if proba > 0.5 else 'LEGIT'} | Confidence: {proba*100:.2f}%")

            elif model_file.name.endswith(".zip"):
                model = DQN.load(model_file)
                obs = np.array(X.iloc[0], dtype=np.float32)
                action, _ = model.predict(obs, deterministic=True)
                st.success(f"Prediction (RL Agent): {'FRAUD' if action==1 else 'LEGIT'}")

            elif model_file.name.endswith(".npy"):
                weights = np.load(model_file)
                scaler = joblib.load("qlearning_scaler.pkl")
                X_scaled = scaler.transform(X)
                score = np.dot(weights, X_scaled[0])
                confidence = 1 / (1 + np.exp(-score))
                label = int(confidence > 0.5)
                st.success(f"Prediction (Q-Learning): {'FRAUD' if label==1 else 'LEGIT'} | Confidence: {confidence*100:.2f}%")
# --- SECTION 5: Batch Prediction Upload ---
st.header("5. Batch Prediction from Excel")
batch_file = st.file_uploader("Upload Excel file for batch prediction", type=['xlsx'])

if batch_file:
    df_batch = pd.read_excel(batch_file)
    st.dataframe(df_batch.head())

    # Apply feature engineering
    def batch_feature_engineering(df):
        df['balance_diff_org'] = df['oldbalanceOrg'] - df['newbalanceOrg']
        df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        df['amount_diff_org'] = df['amount'] - df['balance_diff_org']
        df['amount_diff_dest'] = df['amount'] - df['balance_diff_dest']
        df['txn_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-6)
        df['is_sender_zero_bal'] = (df['oldbalanceOrg'] == 0).astype(int)
        df['is_receiver_zero_before'] = (df['oldbalanceDest'] == 0).astype(int)
        df['is_receiver_exact_amount'] = (df['newbalanceDest'] - df['oldbalanceDest'] == df['amount']).astype(int)
        df['is_large_txn'] = (df['amount'] > 50000).astype(int)
        df['org_to_dest_same'] = (df['nameOrig'].str[0] == df['nameDest'].str[0]).astype(int)
        df['sender_is_customer'] = df['nameOrig'].str.startswith('C').astype(int)
        df['receiver_is_customer'] = df['nameDest'].str.startswith('C').astype(int)
        df['receiver_is_merchant'] = df['nameDest'].str.startswith('M').astype(int)
        df['risk_combo'] = df['is_receiver_zero_before'] & df['is_large_txn'] & df['receiver_is_customer']
        df['is_night'] = (df['step'] % 24 <= 6).astype(int)
        return df

    batch_features = batch_feature_engineering(df_batch.copy())
    batch_engineered = batch_features.drop(columns=['step', 'type', 'nameOrig', 'nameDest'], errors='ignore')

    st.dataframe(batch_engineered.head())

    selected_batch_features = st.multiselect("Select features for batch prediction", list(batch_engineered.columns), default=list(batch_engineered.columns))
    X_batch = batch_engineered[selected_batch_features]

    model_file_batch = st.file_uploader("Upload trained model for batch (.pkl, .h5)", type=["pkl", "h5", "zip", "npy"], key="batch_model")
    if model_file_batch and st.button("Run Batch Prediction"):
        predictions = []

        if model_file_batch.name.endswith(".pkl"):
            model = joblib.load(model_file_batch)
            scaler = joblib.load("fraud_scaler.pkl")
            X_scaled = scaler.transform(X_batch)
            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]
            predictions = pd.DataFrame({"Prediction": preds, "Confidence": probs})

        elif model_file_batch.name.endswith(".h5"):
            model = load_model(model_file_batch)
            scaler = joblib.load("fraud_scaler.pkl")
            X_scaled = scaler.transform(X_batch)
            probs = model.predict(X_scaled).flatten()
            preds = (probs > 0.5).astype(int)
            predictions = pd.DataFrame({"Prediction": preds, "Confidence": probs})

        elif model_file_batch.name.endswith(".npy"):
            weights = np.load(model_file_batch)
            scaler = joblib.load("qlearning_scaler.pkl")
            X_scaled = scaler.transform(X_batch)
            scores = np.dot(X_scaled, weights)
            confidences = 1 / (1 + np.exp(-scores))
            preds = (confidences > 0.5).astype(int)
            predictions = pd.DataFrame({"Prediction": preds, "Confidence": confidences})

        df_result = pd.concat([df_batch.reset_index(drop=True), predictions], axis=1)

        st.subheader("Sample Predictions")
        st.dataframe(df_result.sample(min(50, len(df_result))))

        # Download options
        st.download_button("Download Results as Excel", data=df_result.to_excel(index=False), file_name="batch_predictions.xlsx")
        st.download_button("Download Results as CSV", data=df_result.to_csv(index=False), file_name="batch_predictions.csv")
        st.download_button("Download Results as JSON", data=df_result.to_json(orient="records"), file_name="batch_predictions.json")