import os
import json
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

CURATED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "curated", "faers")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def extract_features(record):
    patient = record.get("patient", {})
    drugs = patient.get("drug", [])
    reactions = patient.get("reaction", [])

    try:
        age = float(patient.get("patientonsetage", 0) or 0)
    except (ValueError, TypeError):
        age = 0.0

    try:
        sex = int(patient.get("patientsex", 0) or 0)
    except (ValueError, TypeError):
        sex = 0

    num_drugs = len(drugs)
    num_reactions = len(reactions)
    serious = 1 if record.get("serious") == "1" else 0

    return [num_drugs, num_reactions, age, sex], serious


def build_dataset():
    X, y = [], []
    files = [f for f in os.listdir(CURATED_DIR) if f.endswith(".json")]
    print(f"Loading {len(files)} FAERS files for feature extraction...")

    for fname in files:
        with open(os.path.join(CURATED_DIR, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        for r in results:
            try:
                features, label = extract_features(r)
                X.append(features)
                y.append(label)
            except Exception:
                continue

    print(f"Built dataset with {len(X)} samples.")
    return X, y


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = build_dataset()

    if len(X) < 20:
        print("Not enough data to train. Exiting.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=["Not Serious", "Serious"]))

    model_path = os.path.join(MODEL_DIR, "severity_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()