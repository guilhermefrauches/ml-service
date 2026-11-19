import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
FEATURES = [
    "idade",
    "renda_mensal",
    "valor_solicitado",
    "num_parcelas",
    "atrasos_anteriores",
    "tempo_emprego_meses",
]
ALVO = "inadimplente"

df = pd.read_csv(BASE / "data" / "dataset.csv")
X = df[FEATURES]
y = df[ALVO]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modelo = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")),
])
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

matriz = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = matriz.ravel()

metricas = {
    "acuracia": round(accuracy_score(y_test, y_pred), 4),
    "precisao": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1": round(f1_score(y_test, y_pred), 4),
    "auc": round(roc_auc_score(y_test, y_proba), 4),
    "verdadeiros_negativos": int(tn),
    "falsos_positivos": int(fp),
    "falsos_negativos": int(fn),
    "verdadeiros_positivos": int(tp),
}

(BASE / "model").mkdir(exist_ok=True)
(BASE / "reports").mkdir(exist_ok=True)

joblib.dump(modelo, BASE / "model" / "modelo.joblib")

with open(BASE / "reports" / "metricas.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)

ConfusionMatrixDisplay(matriz, display_labels=["Adimplente", "Inadimplente"]).plot(cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.savefig(BASE / "reports" / "matriz_confusao.png", bbox_inches="tight")
plt.close()

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("Curva ROC")
plt.savefig(BASE / "reports" / "curva_roc.png", bbox_inches="tight")
plt.close()

print(classification_report(y_test, y_pred, target_names=["Adimplente", "Inadimplente"]))
print(json.dumps(metricas, indent=2, ensure_ascii=False))
