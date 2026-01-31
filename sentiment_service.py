# sentiment_service.py
from typing import List, Dict
import joblib
from pathlib import Path

MODEL_PATH = Path("models/svm_pipeline.pkl")

print("Loading SVM pipeline...")
pipeline = joblib.load(MODEL_PATH)   # pipeline: tfidf + SVM
print("SVM pipeline loaded.")

NEGATIVE_HINT_WORDS = [
    "jelek",
    "buruk",
    "parah",
    "kecewa",
    "benci",
    "sampah",
    "gak jelas",
    "ga jelas",
    "gaje",
    "bohong",
    "tipu",
    "banyak iklan",
    "iklan mulu",
    "iklan terus",
    "ganggu",
    "parah",
    "tolol",
    "monyet",
    "anj",
    "anjing",
    "goblok",
    "aneh",
    "tll",
    "gk guna",
    "tigak guna",
    "jokodok",
    "jokowi",
    "termul",
    "sontoloyo",
]
POSITIVE_HINT_WORDS = [
    "keren",
    "bagus",
    "mantap",
    "mantul",
    "suka",
    "terbaik",
    "luar biasa",
    "recommended",
    "alhamdullilah",
    "amin",
    "ya allah",
    "allah",
    "boleh",
    "okelah",
    "oke",
    "good",
    "mantul",
    "nice",
    "akhirnya",

]

def _apply_heuristic(label: str, text: str, score: float) -> Dict:
    """Koreksi label/score pakai keyword sederhana."""
    lower = text.lower()

    l = label.lower()
    # anggap netral tapi ada kata negatif kuat -> paksa negatif
    if l.startswith("net") or l.startswith("neu"):
        if any(w in lower for w in NEGATIVE_HINT_WORDS):
            return {"sentiment": "Negatif", "score": max(score, 0.75)}
        if any(w in lower for w in POSITIVE_HINT_WORDS):
            return {"sentiment": "Positif", "score": max(score, 0.75)}

    # kalau model sudah negatif/positif, biarkan saja
    # (tapi bisa kamu tambah rule lain kalau mau)
    return {"sentiment": label, "score": score}


def analyze_text(text: str) -> Dict:
    """Prediksi sentimen untuk satu teks komentar."""
    label = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_

    label_index = list(classes).index(label)
    score = float(proba[label_index])

    corrected = _apply_heuristic(str(label), text, score)
    return corrected

def analyze_list(texts: List[str]) -> List[Dict]:
    """Prediksi sentimen untuk list komentar."""
    if not texts:
        return []

    labels = pipeline.predict(texts)
    probas = pipeline.predict_proba(texts)
    classes = list(pipeline.classes_)

    results: List[Dict] = []
    for text, label, proba in zip(texts, labels, probas):
        label_index = classes.index(label)
        score = float(proba[label_index])

        corrected = _apply_heuristic(str(label), text, score)
        results.append({
            "text": text,
            "sentiment": corrected["sentiment"],
            "score": corrected["score"],
        })
    return results
