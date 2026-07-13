import os
import joblib
import torch
import torch.nn as nn

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "models")

_severity_model = None
_phase_model = None
_phase_vectorizer = None
_phase_label_encoder = None


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


def _load_severity_model():
    global _severity_model
    if _severity_model is None:
        path = os.path.join(MODEL_DIR, "severity_model.pkl")
        if os.path.exists(path):
            _severity_model = joblib.load(path)
    return _severity_model


def _load_phase_model():
    global _phase_model, _phase_vectorizer, _phase_label_encoder
    if _phase_model is None:
        config_path = os.path.join(MODEL_DIR, "phase_model_config.pkl")
        weights_path = os.path.join(MODEL_DIR, "phase_classifier.pt")
        vec_path = os.path.join(MODEL_DIR, "phase_vectorizer.pkl")
        enc_path = os.path.join(MODEL_DIR, "phase_label_encoder.pkl")

        if not all(os.path.exists(p) for p in [config_path, weights_path, vec_path, enc_path]):
            return None, None, None

        config = joblib.load(config_path)
        model = PhaseClassifier(config["input_size"], config["hidden_size"], config["num_classes"])
        model.load_state_dict(torch.load(weights_path))
        model.eval()

        _phase_model = model
        _phase_vectorizer = joblib.load(vec_path)
        _phase_label_encoder = joblib.load(enc_path)

    return _phase_model, _phase_vectorizer, _phase_label_encoder


def predict_severity(num_drugs, num_reactions, age, sex):
    model = _load_severity_model()
    if model is None:
        return None
    pred = model.predict([[num_drugs, num_reactions, age, sex]])[0]
    proba = model.predict_proba([[num_drugs, num_reactions, age, sex]])[0]
    confidence = float(max(proba))
    label = "Serious" if pred == 1 else "Not Serious"
    return {"prediction": label, "confidence": round(confidence * 100, 1)}


def predict_trial_phase(text):
    model, vectorizer, label_encoder = _load_phase_model()
    if model is None:
        return None
    X = vectorizer.transform([text]).toarray()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        output = model(X_t)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    label = label_encoder.inverse_transform([pred_idx])[0]
    return {"prediction": str(label), "confidence": round(confidence * 100, 1)}