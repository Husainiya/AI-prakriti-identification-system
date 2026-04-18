import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/balanced_features.csv"
MODEL_PATH = "models/prakriti_model.pkl"


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path):
    print("Loading dataset...")
    df = pd.read_csv(path)
    return df


# -----------------------------
# PREPARE DATA
# -----------------------------
def prepare_data(df):
    print("Preparing features and labels...")
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y


# -----------------------------
# SPLIT DATA
# -----------------------------
def split_data(X, y):
    print("Splitting dataset...")
    return train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model(X_train, y_train):
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# -----------------------------
# EVALUATE MODEL
# -----------------------------
def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Accuracy:", round(accuracy * 100, 2), "%\n")

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))


# -----------------------------
# SAVE MODEL
# -----------------------------
def save_model(model, path):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, path)
    print("\nModel saved at:", path)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    df = load_data(DATA_PATH)

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model, MODEL_PATH)


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    main()