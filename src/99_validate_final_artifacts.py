"""
Validador ampliado dos artefatos finais da decisao.

NAO recalcula metricas; apenas carrega os 3 artefatos finais e faz assertions.

Valida:
  - PRIMARY == apartamento 2Q Meia Praia; N_owners == 112; pickup ~ 1541.91667
  - preços de compra: Morretes 790000, Meia 1080000, Centro 1150000, Centro1 890000
  - BEST_REVENUE_PROFILE: N=45, 31.941, 874
  - pickup textuais: Primary 1.541,92; Centro1 3.469,96
  - redteam 4 critico / 5 importante / 1 secundario; owners host dependence
  - ausencia de termos proibidos
"""

import io
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parent.parent / "outputs" / "tables"

rec = pd.read_csv(OUT / "final_recommendation.csv")
kf = pd.read_csv(OUT / "final_key_findings.csv")
rt = pd.read_csv(OUT / "redteam_risks.csv")


def approx(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol


def r_of(role):
    return rec[rec["role"] == role].iloc[0]


# ---------- PRIMARY ----------
prim = r_of("PRIMARY")
assert prim["segment"] == "Apartamento 2 quartos Meia Praia", prim["segment"]
assert int(prim["N_owners"]) == 112, prim["N_owners"]
assert approx(prim["pickup_median"], 1541.91667), prim["pickup_median"]
assert float(prim["median_purchase_price"]) == 1080000.0, prim["median_purchase_price"]

# ---------- ALTERNATIVES ----------
alt1 = r_of("ALTERNATIVE_1")
assert float(alt1["median_purchase_price"]) == 790000.0, alt1["median_purchase_price"]
assert approx(alt1["bootstrap_median"], 0.01606, tol=2e-4), alt1["bootstrap_median"]

alt2 = r_of("ALTERNATIVE_2")
assert float(alt2["median_purchase_price"]) == 1150000.0, alt2["median_purchase_price"]

cen1 = r_of("INTERNAL_THESIS")
assert float(cen1["median_purchase_price"]) == 890000.0, cen1["median_purchase_price"]
assert approx(cen1["pickup_median"], 3469.95835), cen1["pickup_median"]
assert cen1["pareto_status"] == "DOM", cen1["pareto_status"]

# ---------- perfis ----------
p4 = kf[kf["question"] == "BEST_REVENUE_PROFILE"].iloc[0]
assert p4["answer"] == "Apartamento de 4 quartos", p4["answer"]
assert "N = 45" in str(p4["sample"]), p4["sample"]
assert "31.941" in str(p4["metric_1"]), p4["metric_1"]
assert "874" in str(p4["metric_2"]), p4["metric_2"]

# ---------- pickup textuais ----------
primk = kf[kf["question"] == "PRIMARY_INVESTMENT"].iloc[0]
cen1k = kf[kf["question"] == "CENTRO1_THESIS"].iloc[0]
assert "1.541,92" in str(primk["metric_2"]), primk["metric_2"]
assert "3.469,96" in str(cen1k["metric_2"]), cen1k["metric_2"]

# ---------- redteam ----------
vc = rt["severity"].value_counts().to_dict()
assert vc.get("critico") == 4, vc
assert vc.get("importante") == 5, vc
assert vc.get("secundario") == 1, vc
hd = rt[rt["risk"].str.lower().str.contains("host")].iloc[0]
assert "112" in str(hd["evidence"]), hd["evidence"]

# ---------- bounds ----------
for v in rec["capital_efficiency_91"]:
    assert 0 < float(v) < 1, v

# ---------- termos proibidos ----------
blob = " ".join(str(x) for x in [rec.to_string(), kf.to_string(), rt.to_string()])
forbidden = ["Morrotes", "Morreutes", "pickoup", "fragilis", "linas", "sem imparcial",
             "sinais independentes", "4-5 quartos", "N=53", "R$766", "ROI anual",
             "yield anual", "receita anual"]
for b in forbidden:
    assert b not in blob, b

print("VALIDATOR AMPLIADO OK: todos os artefatos finais passaram.")
print("severidade redteam:", vc)