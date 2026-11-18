"""Classificador por similaridade semântica; não é detector certificado de fraude."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_PATH = Path(__file__).resolve().parent / "data" / "mensagens.csv"
PATTERNS = {
    "Pedido de senha ou código": r"\b(senha|c[oó]digo|cvv)\b",
    "Pressão e urgência": r"\b(urgente|imediatamente|agora|bloquead[ao]|cancelad[ao])\b",
    "Solicitação de pagamento": r"\b(pix|taxa|pagamento|transfira|pague)\b",
    "Link ou endereço": r"\b(link|clique|endere[cç]o)\b",
}

def load_data():
    df = pd.read_csv(DATA_PATH)
    if df.isna().any().any() or not {"texto", "rotulo", "conjunto"}.issubset(df.columns):
        raise ValueError("CSV inválido")
    return df

def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()

def load_model():
    # O modelo é baixado na primeira execução e executado localmente no runtime.
    return SentenceTransformer(MODEL_ID)

def build_index(model, df):
    ref = df[df.conjunto == "referencia"].reset_index(drop=True)
    embeddings = model.encode(ref.texto.map(clean_text).tolist(), normalize_embeddings=True, show_progress_bar=False)
    return ref, np.asarray(embeddings)

def predict(text, model, ref, embeddings, k=3):
    text = clean_text(text)
    if not text:
        raise ValueError("Digite uma mensagem")
    query = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = embeddings @ query
    top = np.argsort(-scores)[:max(1, min(int(k), len(scores)))]
    # Voto ponderado por similaridade não negativa; não é probabilidade calibrada.
    weights = np.maximum(scores[top], 0)
    suspicious = np.array([ref.iloc[i].rotulo == "suspeita" for i in top], dtype=float)
    score = float((weights @ suspicious) / weights.sum()) if weights.sum() > 0 else 0.5
    label = "suspeita" if score >= 0.5 else "legitima"
    signals = [name for name, pattern in PATTERNS.items() if re.search(pattern, text, re.IGNORECASE)]
    examples = [{"texto": str(ref.iloc[i].texto), "rotulo": str(ref.iloc[i].rotulo), "similaridade": round(float(scores[i]), 3)} for i in top]
    return {"rotulo": label, "indice": round(score, 3), "sinais": signals, "exemplos": examples}

def evaluate(model, ref, embeddings, df, k=3):
    test = df[df.conjunto == "teste"].copy()
    truth = test.rotulo.tolist()
    preds = [predict(t, model, ref, embeddings, k)["rotulo"] for t in test.texto]
    return {"accuracy": accuracy_score(truth, preds), "report": classification_report(truth, preds, output_dict=True, zero_division=0), "confusion": confusion_matrix(truth, preds, labels=["legitima", "suspeita"]), "truth": truth, "preds": preds, "test": test}
