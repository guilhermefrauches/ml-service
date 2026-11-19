from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

LIMIAR_CONFIANCA = 0.70
FEATURES = [
    "idade",
    "renda_mensal",
    "valor_solicitado",
    "num_parcelas",
    "atrasos_anteriores",
    "tempo_emprego_meses",
]

modelo = joblib.load(Path(__file__).parent / "model" / "modelo.joblib")

app = FastAPI(title="Motor de Decisão de Crédito")


class Solicitacao(BaseModel):
    idade: int = Field(ge=18, le=100)
    renda_mensal: float = Field(gt=0)
    valor_solicitado: float = Field(gt=0)
    num_parcelas: int = Field(gt=0)
    atrasos_anteriores: int = Field(ge=0)
    tempo_emprego_meses: int = Field(ge=0)


class Decisao(BaseModel):
    classe: str
    probabilidade: float
    aprovacao_manual: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Decisao)
def predict(solicitacao: Solicitacao):
    entrada = pd.DataFrame([solicitacao.model_dump()])[FEATURES]
    probabilidades = modelo.predict_proba(entrada)[0]
    indice = int(probabilidades.argmax())
    probabilidade = float(probabilidades[indice])

    return Decisao(
        classe="negado" if indice == 1 else "aprovado",
        probabilidade=round(probabilidade, 4),
        aprovacao_manual=probabilidade < LIMIAR_CONFIANCA,
    )
