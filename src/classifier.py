from __future__ import annotations
import json, os, re
from typing import Tuple, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

LABELS = ["Urgent", "Normal", "Low"]

def _normalize_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_seed(path: str = "data/seed_samples.jsonl") -> Tuple[List[str], List[int]]:
    X, y = [], []
    if not os.path.exists(path):
        samples = [
            {"subject":"Server down","body":"Checkout failing since midnight", "label":"Urgent"},
            {"subject":"Weekly update","body":"No action required", "label":"Low"},
            {"subject":"Need logs","body":"Please share logs by EOD", "label":"Normal"},
        ]
    else:
        with open(path, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
    for s in samples:
        text = f"{s.get('subject','')}\n\n{s.get('body','')}"
        X.append(_normalize_text(text))
        y.append(LABELS.index(s["label"]))
    return X, y

class EmailPriorityModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1,2),
            min_df=1,
            max_features=8000,
            stop_words="english"
        )
        self.clf = LogisticRegression(max_iter=300, class_weight="balanced")

    def fit(self, texts: List[str], labels: List[int]):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        return self

    def predict(self, subject: str, body: str):
        text = _normalize_text(f"{subject}\n\n{body}")
        X = self.vectorizer.transform([text])
        proba = self.clf.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        return LABELS[idx], float(proba[idx]), proba, self._top_features(X, idx)

    def _top_features(self, X_row, idx: int, k: int = 5) -> List[str]:
        if not hasattr(self.clf, "coef_"):
            return []
        coef = self.clf.coef_
        row = X_row.tocoo()
        weights = {}
        for i, v in zip(row.col, row.data):
            weights[i] = v * coef[idx, i]
        if not weights:
            return []
        top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:k]
        inv_vocab = {v:k for k,v in self.vectorizer.vocabulary_.items()}
        return [inv_vocab[i] for i,_ in top]

def train_or_load() -> EmailPriorityModel:
    X, y = load_seed()
    model = EmailPriorityModel().fit(X, y)
    return model
