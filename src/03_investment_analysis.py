"""
Análise de investimento Airbnb x VivaReal (Itapema).

Métrica principal Airbnb: static_revenue_proxy_91 (H30/H60 = sensibilidades).
Pickup temporal = 2a evidencia independente.
Regras:
  - NÃO anualiza métrica;
  - NÃO trata capital_efficiency como ROI real / yield anual (é proxy bruto p/ janela);
  - NÃO usa ML; NÃO escolhe vencedor antes; NÃO cria score ponderado arbitrário.
  - Ligação Airbnb<->VivaReal é CONCEITUAL (bairro x bedrooms x tipo); não há match real.

Artefatos: outputs/tables/.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False)
    print(f"  saved -> outputs/tables/{name} (rows={len(df)})")


def faixa(n):
    if n < 5:
        return "critico_<5"
    if n < 20:
        return "pequeno_5-19"
    if n < 50:
        return "medio_20-49"
    return "grande_50+"


def norm_bairro(s):
    """Mapping explícito (sem fuzzy matching). Retorna None se não defensável."""
    if s is None or (isinstance(s, float) and (pd.isna(s) or s == "")):
        return None
    s = str(s).strip()
    if not s:
        return None
    k = s.lower()
    mp = {
        "centro": "Centro",
        "meia praia": "Meia Praia",
        "meia praia - frente mar": "Meia Praia",
        "meia praia - frente-mar": "Meia Praia",
        "morretes": "Morretes",
        "tabuleiro": "Tabuleiro dos Oliveiras",
        "taboleiro": "Tabuleiro dos Oliveiras",
        "tabuleiro dos oliveiras": "Tabuleiro dos Oliveiras",
        "sertao do trombudo": "Sertao do Trombudo",
        "sertaozinho": "Sertaozinho",
        "sertao do trombudo": "Sertao do Trombudo",
        "altas sao bento": "Alto Sao Bento",
        "alto sao bento": "Alto Sao Bento",
        "casa branca": "Casa Branca",
        "canto da praia": "Canto da Praia",
        "ilhota": "Ilhota",
        "varzea": "Varzea",
        "jardim praia mar": "Jardim Praiamar",
        "andorinha": "Andorinha",
        "estreito": "Estreito",
        "ocean tower": "Ocean Tower",
    }
    return mp.get(k, s.title())


# ============================================================
# 1. LIMPEZA E AUDITORIA DO VivaReal
# ============================================================
print("=" * 78)
print("1. LIMPEZA E AUDITORIA VivaReal")
vr = pd.read_csv(DATA / "VivaReal_Itapema.csv", dtype={"listing_id": str})
print("  linhas:", len(vr), "| ids unicos:", vr["listing_id"].nunique(),
      "| ocorrencias repetidas:", int(vr["listing_id"].duplicated().sum()))

dup = vr[vr["listing_id"].duplicated(keep=False)]
groups = dup.groupby("listing_id")
all_cols = vr.columns.tolist()
fields = ["sale_price", "usable_area", "bedrooms", "bathrooms", "parking_spaces",
          "suburb", "listing_type", "business_types", "monthly_condo_fee", "yearly_iptu"]
exact = amen_only = other = 0
for gid, g in groups:
    base = g.iloc[0].fillna("__NA__")
    difset = set()
    for _, row in g.iloc[1:].fillna("__NA__").iterrows():
        for c in all_cols:
            if row[c] != base[c]:
                difset.add(c)
    if not difset:
        exact += 1
    elif difset <= {"amenities"}:
        amen_only += 1
    else:
        other += 1
key_consistent = 0
for gid, g in groups:
    if g[fields].fillna("__NA__").nunique().eq(1).all():
        key_consistent += 1
print(f"  grupos dup: {groups.ngroups} | {exact} cópias exatas, "
      f"{amen_only} diferem só em emitters, {other} outras divergências")
print(f"  grupos com campos-chave 100% consistentes: {key_consistent}/{groups.ngroups}")

dups_audit = pd.DataFrame([{
    "linhas": len(vr),
    "listing_id_unicos": vr["listing_id"].nunique(),
    "ocorrencias_repetidas": int(vr["listing_id"].duplicated().sum()),
    "grupos": groups.ngroups,
    "copias_exatas": exact,
    "diferem_so_amenities": amen_only,
    "diferem_outras_colunas": other,
    "campos_chave_consistentes": key_consistent,
}])
save(dups_audit, "vivareal_duplicates_audit.csv")

vr = vr.drop_duplicates("listing_id", keep="first").copy()
print("  apos dedup linhas:", len(vr))

# ============================================================
# 2. UNIVERSO RESIDENCIAL COMPARÁVEL
# ============================================================
print("=" * 78)
print("2. UNIVERSO RESIDENCIAL")
print("  listing_type:", vr["listing_type"].value_counts().to_dict())
vr["is_residential_comparable"] = vr["listing_type"].isin(["apartamento", "casa"])
print("  residencial (apartamen+casa):", int(vr["is_residential_comparable"].sum()))
print("  business_types:", vr["business_types"].value_counts().to_dict())
ambos = vr[vr["business_types"] == "Ambos"]
print("  Ambos: N =", len(ambos), "| com sale_price:", int(ambos["sale_price"].notna().sum()))

# ============================================================
# 3. NORMALIZAÇÃO EXPLÍCITA DE BAIRROS
# ============================================================
print("=" * 78)
print("3. NORMALIZAÇÃO DE BAIRROS")
vr["suburb_original"] = vr["suburb"]
vr["suburb_canonical"] = vr["suburb"].apply(norm_bairro)
vr["flag_frente_mar"] = vr["suburb"].astype(str).str.lower().str.contains("frente mar").astype(int)
vr["suburb_mapping_applied"] = vr["suburb_canonical"].notna().astype(int)
submap = (vr[["suburb_original", "suburb_canonical"]].value_counts()
          .rename("n").reset_index().sort_values("n", ascending=False))
save(submap, "vivareal_suburb_mapping.csv")
print("  mapeados:", int(vr["suburb_mapping_applied"].sum()),
      "| NA (sem direto):", int((vr["suburb_canonical"].isna()).sum()))
print(submap.to_string(index=False))

# ============================================================
# 4. OUTLIERS — não remover silenciosamente
# ============================================================
print("=" * 78)
print("4. OUTLIERS")
vr["sale_price_num"] = pd.to_numeric(vr["sale_price"], errors="coerce")
vr["usable_area_num"] = pd.to_numeric(vr["usable_area"], errors="coerce")
vr["price_per_m2"] = np.where(vr["usable_area_num"] > 0,
                              vr["sale_price_num"] / vr["usable_area_num"], np.nan)
print("  sale_price missing:", int(vr["sale_price_num"].isna().sum()),
      "| usable_area missing:", int(vr["usable_area_num"].isna().sum()),
      "| area<=0 (sem preço/m2):", int((vr["usable_area_num"] <= 0).sum()))
q = vr["sale_price_num"].describe()
print("  sale_price describe:\n", q.round(0))
iqr = q["75%"] - q["25%"]
vr["flag_outlier_price"] = (vr["sale_price_num"] > q["75%"] + 3 * iqr).astype(int)
pm2 = vr["price_per_m2"].quantile(0.99)
vr["flag_outlier_m2"] = (vr["price_per_m2"] > pm2).astype(int)
print("  flag_price_outlier:", int(vr["flag_outlier_price"].sum()),
      "| flag_m2_outlier:", int(vr["flag_outlier_m2"].sum()))
ol = vr[["listing_id", "suburb_original", "suburb_canonical", "sale_price_num",
         "usable_area_num", "price_per_m2", "listing_type", "bedrooms",
         "flag_outlier_price", "flag_outlier_m2"]].sort_values("sale_price_num", ascending=False)
save(ol, "vivareal_outlier_review.csv")

# ============================================================
# 5. SEGMENTOS DE COMPRA (VivaReal)
# ============================================================
print("=" * 78)
print("5. SEGMENTOS DE COMPRA")


def seg_purchase(df):
    out = []
    for (subb, b, tp), g in df.groupby(["suburb_canonical", "bedrooms", "listing_type"], dropna=False):
        n = len(g)
        sp = g["sale_price_num"].dropna()
        rec = {"suburb_canonical": subb, "bedrooms": b, "listing_type": tp, "N_sale": n}
        rec["median_sale_price"] = sp.median() if len(sp) else np.nan
        rec["mean_sale_price"] = sp.mean() if len(sp) else np.nan
        rec["p25_sale_price"] = sp.quantile(0.25) if len(sp) else np.nan
        rec["p75_sale_price"] = sp.quantile(0.75) if len(sp) else np.nan
        rec["median_usable_area"] = g["usable_area_num"].median()
        rec["median_price_per_m2"] = g["price_per_m2"].median()
        cf = pd.to_numeric(g["monthly_condo_fee"], errors="coerce")
        rec["median_monthly_condo_fee"] = cf.median()
        rec["missing_rate_condo"] = cf.isna().mean()
        ip = pd.to_numeric(g["yearly_iptu"], errors="coerce")
        rec["median_yearly_iptu"] = ip.median()
        rec["missing_rate_iptu"] = ip.isna().mean()
        rec["faixa_vivareal"] = faixa(n)
        out.append(rec)
    return pd.DataFrame(out)


purch = seg_purchase(vr)
save(purch, "purchase_segments.csv")

# ============================================================
# 6. INTEGRAR COM AIRBNB (conceitu­al)
# ============================================================
print("=" * 78)
print("6. INTEGRAR COM AIRB")
metrics = pd.read_csv(OUT / "listing_metrics.csv", dtype={"airbnb_listing_id": str})
metrics["suburb_canonical"] = metrics["suburb"].apply(norm_bairro)
metrics["bedrooms"] = pd.to_numeric(metrics["number_of_bedrooms"], errors="coerce")
metrics["listing_type"] = metrics["listing_type"].astype(str).str.strip().str.lower()
bo_bit = metrics[metrics["n_prices_observed_H_91"].fillna(0) > 0].copy()

pick = pd.read_csv(OUT / "pickup_metrics.csv", encoding="utf-8", dtype={"airbnb_listing_id": str})
pick = pick.merge(metrics[["airbnb_listing_id", "suburb_canonical", "bedrooms", "listing_type"]],
                  on="airbnb_listing_id", how="left")

# aggregation helper
purch_idx = purch.set_index(["suburb_canonical", "bedrooms", "listing_type"])

def airbnb_stats(df):
    out = {}
    for (subb, b, tp), g in df.groupby(["suburb_canonical", "bedrooms", "listing_type"], dropna=False):
        g = g[g["n_prices_observed_H_91"].fillna(0) > 0]
        n = len(g)
        out[(subb, b, tp)] = {
            "N_airbnb_91": n,
            "ADR_91_median": g["median_available_price_H_91"].median() if n else np.nan,
            "unavailability_91_median": g["unavailability_rate_H_91"].median() if n else np.nan,
            "static_rev_91_median": g["static_revenue_proxy_H_91"].median() if n else np.nan,
            "static_rev_91_p25": g["static_revenue_proxy_H_91"].quantile(0.25) if n else np.nan,
            "static_rev_91_p75": g["static_revenue_proxy_H_91"].quantile(0.75) if n else np.nan,
            "static_rev_30_median": g["static_revenue_proxy_H_30"].median() if n else np.nan,
            "static_rev_60_median": g["static_revenue_proxy_H_60"].median() if n else np.nan,
            "ADR_30_median": g["median_available_price_H_30"].median() if n else np.nan,
            "ADR_60_median": g["median_available_price_H_60"].median() if n else np.nan,
        }
    return out


def pickup_stats(df):
    out = {}
    for (subb, b, tp), g in df.groupby(["suburb_canonical", "bedrooms", "listing_type"], dropna=False):
        n = len(g)
        out[(subb, b, tp)] = {
            "N_pickup": n,
            "net_pickup_value_median": g["net_pickup_value_proxy"].median() if n else np.nan,
            "net_transition_rate_median": g["net_transition_rate"].median() if n else np.nan,
        }
    return out


ab = airbnb_stats(bo_bit)
pk = pickup_stats(pick)

inv_rows = []
for (subb, b, tp), pr in purch_idx.iterrows():
    key = (subb, b, tp)
    a = ab.get(key, {"N_airbnb_91": 0})
    d = pk.get(key, {"N_pickup": 0, "net_pickup_value_median": np.nan,
                     "net_transition_rate_median": np.nan})
    inv_rows.append({
        "suburb": subb, "bedrooms": b, "listing_type": tp,
        **pr.to_dict(), **a, **d,
    })
inv = pd.DataFrame(inv_rows)

# ============================================================
# 7-8. CAPITAL EFFICIENCY PROXY + EVIDENCE TIER
# ============================================================
print("=" * 78)
print("7-8. CAPITAL EFFICIENCY PROXY + EVIDENCE TIER")
inv["capital_efficiency_proxy_91"] = inv["static_rev_91_median"] / inv["median_sale_price"]
inv["capital_efficiency_proxy_30"] = inv["static_rev_30_median"] / inv["median_sale_price"]
inv["capital_efficiency_proxy_60"] = inv["static_rev_60_median"] / inv["median_sale_price"]
inv["conservative_91"] = inv["static_rev_91_p25"] / inv["p75_sale_price"]
inv["base_91"] = inv["static_rev_91_median"] / inv["median_sale_price"]
inv["optimistic_91"] = inv["static_rev_91_p75"] / inv["p25_sale_price"]
inv["faixa_airbnb"] = inv["N_airbnb_91"].apply(faixa)
inv["faixa_vivareal"] = inv["N_sale"].apply(faixa)
inv["evidence_tier"] = inv.apply(
    lambda r: ("forte" if (r["N_airbnb_91"] >= 50 and r["N_sale"] >= 50)
               else ("moderada" if (r["N_airbnb_91"] >= 20 and r["N_sale"] >= 20)
                     else "exploratoria")), axis=1)
save(inv, "investment_segments.csv")

# ============================================================
# 9. SHORTLIST
# ============================================================
print("=" * 78)
print("9. SHORTLIST (N_airbnb>=20 & N_sale>=20)")
sl_cols = ["suburb", "bedrooms", "listing_type", "N_airbnb_91", "N_sale",
           "ADR_91_median", "static_rev_91_median", "median_sale_price",
           "capital_efficiency_proxy_91",
           "conservative_91", "base_91", "optimistic_91",
           "N_pickup", "net_pickup_value_median", "net_transition_rate_median", "evidence_tier"]
short = inv[(inv["N_airbnb_91"] >= 20) & (inv["N_sale"] >= 20)].sort_values(
    "capital_efficiency_proxy_91", ascending=False).copy()
save(short[sl_cols], "investment_shortlist.csv")
print("  shortlist segments:", len(short))

# ============================================================
# 10. TESE INTERNA — 1Q Centro vs elegíveis
# ============================================================
print("=" * 78)
print("10. TESE INTERNA (1Q Centro)")
alvos = [("apartamento", 1, "Centro"), ("apartamento", 2, "Centro"),
         ("apartamento", 1, "Meia Praia"), ("apartamento", 2, "Meia Praia"),
         ("apartamento", 1, "Morretes"), ("apartamento", 2, "Morretes"),
         ("apartamento", 2, "Tabuleiro dos Oliveiras")]
thesis = []
inv_idx = inv.set_index(["listing_type", "bedrooms", "suburb"])
for (tp, b, sub) in alvos:
    if (tp, b, sub) in inv_idx.index:
        r = inv_idx.loc[(tp, b, sub)]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        thesis.append({
            "perfil": f"{tp} {b}Q {sub}",
            "N_airbnb": r["N_airbnb_91"], "N_vivareal": r["N_sale"],
            "N_pickup": r["N_pickup"], "static_rev_91": r["static_rev_91_median"],
            "median_sale": r["median_sale_price"],
            "cap_eff_91": r["capital_efficiency_proxy_91"],
            "net_pickup_value": r["net_pickup_value_median"],
            "evidence_tier": r["evidence_tier"],
        })
thesis_df = pd.DataFrame(thesis)
# studio+Centro (evidência insuficiente se sem Price_AV)
thesis_df = pd.concat([thesis_df, pd.DataFrame([{
        "perfil": "studio+Centro", "N_airbnb": 0, "N_vivareal": np.nan, "N_pickup": 0,
        "static_rev_91": np.nan, "median_sale": np.nan, "cap_eff_91": np.nan,
        "net_pickup_value": np.nan, "evidence_tier": "evidência_insuficiente"}])],
        ignore_index=True)
save(thesis_df, "thesis_return_comparison.csv")
print("  tese salva em thesis_return_comparison.csv")
print(thesis_df.to_string(index=False))

# ============================================================
# 11. SANITY CHECKS (vs auditoria)
# ============================================================
print("=" * 78)
print("11. SANITY CHECKS (apartamento)")
sanity = {
    ("Centro", 1): (890, 22), ("Centro", 2): (1150, 93),
    ("Meia Praia", 1): (877.5, 58), ("Meia Praia", 2): (1080.0, 243),
    ("Morretes", 1): (624.5, 50), ("Morretes", 2): (790.0, 1037),
    ("Tabuleiro dos Oliveiras", 2): (780.0, 110),
}
for (b, q), (m_aud, n_aud) in sanity.items():
    r = inv[(inv["suburb"] == b) & (inv["bedrooms"] == q) & (inv["listing_type"] == "apartamento")]
    if len(r):
        rr = r.iloc[0]
        print(f"  {b} {q}Q: auditoria N≈{n_aud} med≈{m_aud*1000:,.0f} | nosso N={int(rr['N_sale'])} med={rr['median_sale_price']:,.0f}")

print("\nDONE")