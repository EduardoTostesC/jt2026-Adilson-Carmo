"""
Etapa 06 — DECISAO FINAL ANALITICA (REPRODUTIVEL).

Fonte unica dos 3 artefatos finais:
  outputs/tables/final_recommendation.csv
  outputs/tables/final_key_findings.csv
  outputs/tables/redteam_risks.csv

Decisao locked (nao altera metricas, nao usa dados externos):
  PRIMARY        = Apartamento 2Q Meia Praia
  ALTERNATIVE_1  = Apartamento 2Q Morretes
  ALTERNATIVE_2  = Apartamento 2Q Centro
  INTERNAL_THESIS= Centro 1Q nao sustentado; Studio inconclusivo.

Fluxo: carregar outputs 01-05; assertions; gerar os 3 CSVs;
validar; terminar. Sem pos-processamento.
"""

import io
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# 1. Carrega outputs auditados
# ---------------------------------------------------------------
fin = pd.read_csv(OUT / "final_decision_evidence.csv").set_index("candidato")
boot = pd.read_csv(OUT / "bootstrap_efficiency.csv").set_index("candidato")
inv = pd.read_csv(OUT / "investment_segments.csv").set_index(
    ["listing_type", "bedrooms", "suburb"])
lm = pd.read_csv(OUT / "listing_metrics.csv", dtype={"airbnb_listing_id": str})

CAND = ["apartamento 2Q Meia Praia", "apartamento 2Q Morretes",
        "apartamento 2Q Centro", "apartamento 1Q Centro"]
LABEL = {
    "apartamento 2Q Meia Praia": "Apartamento 2 quartos Meia Praia",
    "apartamento 2Q Morretes": "Apartamento 2 quartos Morretes",
    "apartamento 2Q Centro": "Apartamento 2 quartos Centro",
    "apartamento 1Q Centro": "Apartamento 1 quarto Centro",
}
KEY = {
    "apartamento 2Q Meia Praia": ("apartamento", 2, "Meia Praia"),
    "apartamento 2Q Morretes": ("apartamento", 2, "Morretes"),
    "apartamento 2Q Centro": ("apartamento", 2, "Centro"),
    "apartamento 1Q Centro": ("apartamento", 1, "Centro"),
}

# ---------------------------------------------------------------
# 2. Assertions criticos
# ---------------------------------------------------------------
assert int(boot.loc["apartamento 2Q Meia Praia", "N_owners"]) == 112

apx = lm[(lm["listing_type"].astype(str).str.strip().str.lower() == "apartamento") &
         (pd.to_numeric(lm["number_of_bedrooms"], errors="coerce") == 4) &
         (lm["n_prices_observed_H_91"].fillna(0) > 0)]
assert len(apx) == 45, f"4Q N={len(apx)}"
assert round(float(apx["static_revenue_proxy_H_91"].median()), 0) == 31941
assert round(float(apx["median_available_price_H_91"].median()), 0) == 874

prices = {c: float(inv.loc[KEY[c], "median_sale_price"]) for c in CAND}
assert prices["apartamento 2Q Morretes"] == 790000.0
assert prices["apartamento 1Q Centro"] == 890000.0
assert abs(float(boot.loc["apartamento 2Q Morretes", "cluster_median"]) - 0.01606) < 0.0002
for c in CAND:
    ce = float(inv.loc[KEY[c], "capital_efficiency_proxy_91"])
    assert 0 < ce < 1, c
print("  [2] assertions criticos ok")

# ---------------------------------------------------------------
# 3. final_recommendation.csv
# ---------------------------------------------------------------
def mk_rec(seg, role, decision, strength, risk, rationale):
    ir = inv.loc[KEY[seg]]
    b = boot.loc[seg]
    f = fin.loc[seg]
    return {
        "role": role, "segment": LABEL[seg], "decision": decision,
        "N_airbnb": int(f["N_airbnb"]), "N_owners": int(f["N_owner_airbnb"]),
        "N_sale": int(f["N_sale"]),
        "median_purchase_price": float(ir["median_sale_price"]),
        "static_revenue_proxy_91": float(ir["static_rev_91_median"]),
        "capital_efficiency_91": float(ir["capital_efficiency_proxy_91"]),
        "rank_H30": int(f["rank_H30"]), "rank_H60": int(f["rank_H60"]),
        "rank_H91": int(f["rank_H91"]),
        "pickup_median": float(f["pickup_median"]),
        "bootstrap_median": float(b["cluster_median"]),
        "bootstrap_P2_5": float(b["cluster_P2_5"]),
        "bootstrap_P97_5": float(b["cluster_P97_5"]),
        "partial_cost_eff": float(f["partial_cost_eff"]),
        "pareto_status": f["pareto_status"], "evidence_tier": f["evidence_tier"],
        "main_strength": strength, "main_risk": risk, "rationale": rationale,
    }


DEFS = [
    ("apartamento 2Q Meia Praia", "PRIMARY", "COMPRAR",
     "Convergencia entre multiplas perspectivas complementares: performance estatica "
     "(static revenue proxy R$15.925), dinamica temporal (pickup mediano R$1.541,92) e "
     "robustez amostral (126 listings / 112 owners).",
     "Faixa amostral forte, apesar de cobertura Price_AV snapshot de 17,4%; "
     "preco de compra R$1.080.000 (nao e o mais barato).",
     "Decisao empresarial pela convergencia estatico + dinamico + amostral, com Pareto "
     "nao dominado; o perfil mais defensavel para compra."),
    ("apartamento 2Q Morretes", "ALTERNATIVE_1", "COMPRAR (alternativa)",
     "Maior eficiencia de capital H91 (~1,60%) e menor preco de compra (R$790.000); "
     "Pareto nao dominado.",
     "Faixa Air moderada (43 listings / 34 owners); net pickup value mediano = R$0 na "
     "janela observada de 06/01 a 20/01 (nao implica ausencia total de demanda).",
     "Foco em custo/eficiencia maxima; alternativa por menor robustez amostral e ausencia "
     "de momentum observado na janela."),
    ("apartamento 2Q Centro", "ALTERNATIVE_2", "COMPRAR (alternativa)",
     "Melhor momentum/dinamica (pickup mediano R$3.982,58; transition 8,4%); faixa forte.",
     "Menor eficiencia de capital H91 (~1,30%) e preco mais alto (R$1.150.000).",
     "Para quem priorizar demanda dinamica sobre eficiencia estatica."),
    ("apartamento 1Q Centro", "INTERNAL_THESIS", "NAO RECOMENDADO COMO PRINCIPAL",
     "Cobertura Price_AV elevada; pickup mediano R$3.469,96; preco abaixo de Meia e Centro.",
     "4o em H30/H60/H91; eficiencia H91 ~1,17%; Pareto dominado; N_sale=22; 17 owners.",
     "Tese: Studio + Centro inconclusivo por evidencia insuficiente; "
     "Centro 1Q nao sustentado como alternativa de maior eficiencia de capital."),
]
rec_df = pd.DataFrame([mk_rec(*d) for d in DEFS])
rec_df.to_csv(OUT / "final_recommendation.csv", index=False)
print("  [3] final_recommendation.csv ok")

def format_brl(value):
    """Formata valor BRL com centavos, ex.: 1541.91667 -> 'R$ 1.541,92'."""
    return "R$ {:,.2f}".format(float(value)).replace(",", "X").replace(".", ",").replace("X", ".")


def format_pct(value):
    """Formata fracao decimal como percentual, ex.: 0.014745 -> '1,47%'."""
    return "{:.2f}%".format(float(value) * 100).replace(".", ",")


# pickup derivado das fontes (nunca duplicado manualmente)
pk_brl = {
    "Meia": format_brl(fin.loc["apartamento 2Q Meia Praia", "pickup_median"]),
}
pk_alt2 = format_brl(fin.loc["apartamento 2Q Centro", "pickup_median"])
pk_c1 = format_brl(fin.loc["apartamento 1Q Centro", "pickup_median"])

assert pk_brl["Meia"] == "R$ 1.541,92", pk_brl["Meia"]
assert pk_c1 == "R$ 3.469,96", pk_c1
assert pk_alt2 == "R$ 3.982,58", pk_alt2
print("  [helpers] pickup formatados OK:", pk_brl["Meia"], pk_alt2, pk_c1)


# ---------------------------------------------------------------
# 4. final_key_findings.csv
# ---------------------------------------------------------------
KF = [
    ("BEST_REVENUE_PROFILE", "Apartamento de 4 quartos",
     "static revenue proxy H91 mediana ~ R$ 31.941", "ADR mediana ~ R$ 874",
     "N = 45", "maior perfil combinado com amostra defensavel; receita absoluta != eficiencia"),
    ("BEST_REVENUE_LOCATION", "Meia Praia",
     "H30 ~ R$ 12.874; H60 ~ R$ 18.850; H91 ~ R$ 19.200",
     "Centro lidera momentum por listing, nao static revenue",
     "N H91 Meia Praia ~ 483",
     "bairros com N muito pequeno nao sao robustos"),
    ("TOP_REVENUE_DRIVERS", "Associacao descritiva (nao causal)",
     "quartos/banheiros/guests associados a maior ADR; cleaning fee associado",
     "n_reviews e fotos associados a maior unavailability proxy",
     "N com precos = 777",
     "GroupKFold por owner mostrou baixa generalizacao; associativo, nao predicao"),
("PRIMARY_INVESTMENT", "Apartamento 2 quartos em Meia Praia",
     "capital efficiency H91 ~ 1,47%; static proxy ~ R$ 15.925",
     f"pickup mediano {pk_brl['Meia']}; Pareto nao dominado; rank H60=1, H30/H91=2",
     "N Airbnb=126; owners=112; N venda=243",
     "faixa amostral forte, apesar de cobertura Price_AV snapshot ~17,4%"),
    ("ALTERNATIVE_1", "Apartamento 2 quartos em Morretes",
     "capital efficiency H91 ~ 1,60%", "preco de compra ~ R$ 790.000 (menor)",
     "N Airbnb=43; owners=34; N venda=1037",
     "maior eficiencia e menor preco, mas faixa amostral moderada; pickup mediano R$0"),
    ("ALTERNATIVE_2", "Apartamento 2 quartos em Centro",
     "capital efficiency H91 ~ 1,30%", f"{pk_alt2} pickup mediano; melhor momentum",
     "N Airbnb=59; owners=37; N venda=89",
     "melhor dinamica, porem menor eficiencia estatica e preco mais alto"),
    ("STUDIO_THESIS", "INCONCLUSIVO POR EVIDENCIA INSUFICIENTE",
     "3 studio no Centro no Details", "0 studios com Price_AV",
     "N=0 (com preco)", "sem amostra de receita para concluir"),
    ("CENTRO1_THESIS", "NAO SUSTENTADO COMO ALTERNATIVA DE MAIOR EFICIENCIA DE CAPITAL",
     "cap_eff H ~ 1,17%; 4o em H30/H60/H91; Pareto dominado",
     f"a favor: coverage alta (64,7%), pickup {pk_c1}",
     "N Airbnb=75; owners=17; N venda=22",
     "contra: eficiencia, Pareto e amostra pequena"),
    ("MAIN_LIMITATION", "Tudo e proxy (receita/ocupacao real nao observada)",
     "static = mediana preco x noites indisponiveis",
     "coverage snapshot: Centro1Q 64,7% vs Meia2 17,4%",
     "pool Price_AV ~ 22%",
     "nao anualizar; custos operacionais completos nao observados; semantica da indisponibilidade"),
]
kf_df = pd.DataFrame(KF, columns=["question", "answer", "metric_1", "metric_2",
                                  "sample", "confidence_or_limitation"])
kf_df.to_csv(OUT / "final_key_findings.csv", index=False)
print("  [4] final_key_findings.csv ok")

# ---------------------------------------------------------------
# 5. redteam_risks.csv
# ---------------------------------------------------------------
RT = [
    (1, "Price_AV semantics", "critico",
     "Linha tratada como data disponivel e inferencia; janela D..D+90 com timestamps disjuntos.",
     "Pode inflar ou reduzir a proxy estatica e o ranking.", "rotula proxy; reportar cobertura.", "sim"),
    (2, "unavailable != booking", "critico",
     "Ausencia de data pode ser bloqueio, manutencao, uso do proprietario.",
     "Superestima demanda em bairros que fecham calendario.", "proxy de demanda sempre.", "sim"),
    (3, "Price_AV selection bias", "critico",
     "Coverage snapshot: Centro1Q 64,7%, Centro2Q 32,2%, Meia2 17,4%, Morretes2 18,8%.",
     "Comparacoes podem refletir disponibilidade de dados.", "reportar coverage; nao ponderar.", "sim"),
    (4, "seasonality", "importante",
     "Janela cobre verao/inicio do ano (jan-abr).",
     "Eficiencia pode nao representar o ano.", "rotulo observed-window; nunca x4.", "sim"),
    (5, "VivaReal asking price != transaction", "importante",
     "Preco de anuncio nao transacionado; amostra pequena (ex Centro1 N=22).",
     "Eficiencia pode divergir do proxy.", "usar mediana/intervalo; reconhecer asking.", "sim"),
    (6, "unobserved costs", "critico",
     "Partial-cost cobre so condominio e IPTU.",
     "Pode superestimar retorno liquido.", "nao promover como liquido.", "sim"),
    (7, "airbnb/vivareal conceptual matching", "secundario",
     "Join por bairro x quartos x tipo.",
     "Valido no agregado, nao individual.", "decidir por segmento.", "sim"),
    (8, "host dependence", "importante",
     "N owners H91: Morretes2 34, MeiaPraia2 112, Centro2 37, Centro1 17.",
     "Bootstrap i.i.d. superestima confianca.", "cluster bootstrap; reportar N owners.", "sim"),
    (9, "sample size", "importante",
     "N_sale reduzidos (ex Centro1 N=22) e N Airbnb 43-126.",
     "Mediana instavel; segmentos pequenos frageis.", "exigir minimo; reportar evidencia.", "sim"),
    (10, "static revenue proxy construction", "importante",
     "Estatica = mediana preco x noites indisponiveis; RF nao generaliza.",
     "NAO tratar como predicao forte nem receita real.", "rotular proxy estatica.", "sim"),
]
rt_df = pd.DataFrame(RT, columns=["id", "risk", "severity", "evidence",
                                  "how_it_could_change_decision", "mitigation", "remains_open"])
rt_df.to_csv(OUT / "redteam_risks.csv", index=False)
print("  [5] redteam_risks.csv ok")

print("DONE 06")