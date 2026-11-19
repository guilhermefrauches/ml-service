from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

idade = rng.integers(18, 70, n)
renda_mensal = rng.normal(4000, 1800, n).clip(1200, 20000).round(2)
valor_solicitado = rng.normal(8000, 5000, n).clip(500, 50000).round(2)
num_parcelas = rng.choice([6, 12, 18, 24, 36, 48], n)
atrasos_anteriores = rng.poisson(0.8, n)
tempo_emprego_meses = rng.integers(0, 240, n)

parcela = valor_solicitado / num_parcelas
comprometimento = parcela / renda_mensal

risco = (
    -2.0
    + 6.0 * comprometimento
    + 0.9 * atrasos_anteriores
    - 0.008 * tempo_emprego_meses
    - 0.02 * (idade - 18)
    + rng.normal(0, 0.8, n)
)
inadimplente = (1 / (1 + np.exp(-risco)) > 0.5).astype(int)

df = pd.DataFrame({
    "idade": idade,
    "renda_mensal": renda_mensal,
    "valor_solicitado": valor_solicitado,
    "num_parcelas": num_parcelas,
    "atrasos_anteriores": atrasos_anteriores,
    "tempo_emprego_meses": tempo_emprego_meses,
    "inadimplente": inadimplente,
})

saida = Path(__file__).parent / "dataset.csv"
df.to_csv(saida, index=False)
print(f"{len(df)} registros salvos em {saida}")
print(df["inadimplente"].value_counts())
