import os
import json
import joblib
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CURATED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "curated", "clinical_trials")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
VOCAB_SIZE = 1000
HIDDEN_SIZE = 64


class PhaseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def build_dataset():
    texts, labels = [], []
    files = [f for f in os.listdir(CURATED_DIR) if f.endswith(".json")]
    print(f"Loading {len(files)} clinical trial files...")

    for fname in files:
        try:
            with open(os.path.join(CURATED_DIR, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            protocol = data.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            design = protocol.get("designModule", {})
            desc = protocol.get("descriptionModule", {})

            phases = design.get("phases", [])
            if not phases:
                continue
            phase = phases[0]

            title = ident.get("briefTitle", "")
            summary = desc.get("briefSummary", "")
            text = f"{title} {summary}"

            if text.strip() and phase:
                texts.append(text)
                labels.append(phase)
        except Exception:
            continue

    print(f"Built dataset with {len(texts)} samples.")
    return texts, labels


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    texts, labels = build_dataset()

    if len(texts) < 20:
        print("Not enough data to train. Exiting.")
        return

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    print(f"Classes: {list(label_encoder.classes_)}")

    vectorizer = CountVectorizer(max_features=VOCAB_SIZE, stop_words="english")
    X = vectorizer.fit_transform(texts).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = PhaseClassifier(input_size=X_train.shape[1], hidden_size=HIDDEN_SIZE, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 15
    print("\nTraining neural network...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_t)
                test_preds = torch.argmax(test_outputs, dim=1)
                acc = (test_preds == y_test_t).float().mean().item()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Test Accuracy: {acc:.3f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "phase_classifier.pt"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "phase_vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "phase_label_encoder.pkl"))
    joblib.dump({"input_size": X_train.shape[1], "hidden_size": HIDDEN_SIZE, "num_classes": num_classes},
                os.path.join(MODEL_DIR, "phase_model_config.pkl"))

    print(f"\nModel and vectorizer saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()