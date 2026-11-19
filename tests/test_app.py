import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

BOM_PAGADOR = {
    "idade": 45,
    "renda_mensal": 9000,
    "valor_solicitado": 3000,
    "num_parcelas": 24,
    "atrasos_anteriores": 0,
    "tempo_emprego_meses": 120,
}

MAU_PAGADOR = {
    "idade": 20,
    "renda_mensal": 1500,
    "valor_solicitado": 30000,
    "num_parcelas": 6,
    "atrasos_anteriores": 5,
    "tempo_emprego_meses": 2,
}


def test_health():
    resposta = client.get("/health")
    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


def test_predict_retorna_campos():
    resposta = client.post("/predict", json=BOM_PAGADOR)
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"classe", "probabilidade", "aprovacao_manual"}
    assert 0 <= corpo["probabilidade"] <= 1


def test_bom_pagador_aprovado():
    corpo = client.post("/predict", json=BOM_PAGADOR).json()
    assert corpo["classe"] == "aprovado"
    assert corpo["aprovacao_manual"] is False


def test_mau_pagador_negado():
    corpo = client.post("/predict", json=MAU_PAGADOR).json()
    assert corpo["classe"] == "negado"


def test_aprovacao_manual_segue_limiar():
    corpo = client.post("/predict", json=BOM_PAGADOR).json()
    assert corpo["aprovacao_manual"] == (corpo["probabilidade"] < 0.70)


def test_payload_invalido():
    resposta = client.post("/predict", json={**BOM_PAGADOR, "idade": 10})
    assert resposta.status_code == 422
