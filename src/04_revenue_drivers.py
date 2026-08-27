"""
Etapa 4 — EXPLICABILIDADE dos drivers de receita (versão rigorosa / PATCH final).

"Quais características explicam as melhores receitas?" — ASSOCIAÇÃO, NÃO causalidade.
Prioriza interpretabilidade; não maximiza performance de decisão.
Não altera data/, não recalcula scripts 01/02/03, não faz recomendação final.

Rigida metodológica:
  - ratings/satisfaction zero + sem reviews => NaN (colunas *_clean);
  - amenities parsing auditado (falha != 0);
  - CV principal por owner (GroupKFold); KFold só como sensibilidade;
  - baseline DummyRegressor(median) nas MESMAS folds;
  - permutation importance via Pipeline no espaço original, split por host;
  - linguagem de associação (não causal).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "tables"
FIG = Path(__file__).resolve().parent.parent / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
RS = 42


def save(df, name):
    df.to_csv(OUT / name, index=False)
    print("  saved:", name, "(", len(df), "linhas)")


def faixa(n):
    return ("critico_<5" if n < 5 else "pequeno_5-19" if n < 20
            else "medio_20-49" if n < 50 else "grande_50+")


def parse_amen(s):
    """Retorna int len se precisava; None p/ missing; -1 p/ falha de parse."""
    if pd.isna(s):
        return None
    try:
        return len(json.loads(s))
    except Exception:
        return -1


def to_bool(s):
    s = str(s).strip().lower()
    return 1 if s == "true" else (0 if s == "false" else np.nan)


print("=" * 78)
print("1. BASE DE FEATURES")
details = pd.read_csv(DATA / "Details_Itapema.csv", dtype={"airbnb_listing_id": str, "owner_id": str}, low_memory=False)
hosts = pd.read_csv(DATA / "Hosts_ids_Itapema.csv", dtype={"owner_id": str}, low_memory=False)
mesh = pd.read_csv(DATA / "Mesh_Ids_Data_Itapema.csv", dtype={"airbnb_listing_id": str})
metrics = pd.read_csv(OUT / "listing_metrics.csv", dtype={"airbnb_listing_id": str})
inv = pd.read_csv(OUT / "investment_segments.csv")

assert details["airbnb_listing_id"].is_unique
assert mesh["airbnb_listing_id"].is_unique
hosts["host_snapshot_date"] = pd.to_datetime(hosts["host_snapshot_date"], errors="coerce")
hosts_latest = hosts.sort_values("host_snapshot_date").drop_duplicates("owner_id", keep="last").copy()
assert hosts_latest["owner_id"].is_unique
print("  hosts_latest:", len(hosts_latest))

base = details.merge(mesh[["airbnb_listing_id", "suburb"]], on="airbnb_listing_id", how="inner")
base = base.merge(hosts_latest, on="owner_id", how="left")
base = base.merge(metrics, on="airbnb_listing_id", how="left", suffixes=("", "_m"))
assert base["airbnb_listing_id"].is_unique
print("  base linhas:", len(base))

base["A_static_rev"] = pd.to_numeric(base["static_revenue_proxy_H_91"], errors="coerce")
base["B_adr"] = pd.to_numeric(base["median_available_price_H_91"], errors="coerce")
base["C_unavail"] = pd.to_numeric(base["unavailability_rate_H_91"], errors="coerce")

# ============================================================
# 1. RATINGS / SATISFACTION CLEAN
# ============================================================
print("=" * 78)
print("1. ratings/satisfaction CLEAN")
for c in ["number_of_reviews", "star_rating", "guest_satisfaction_overall"]:
    base[c] = pd.to_numeric(base[c], errors="coerce")
r0 = base["number_of_reviews"].fillna(-1) == 0
s0 = base["star_rating"].fillna(-1) == 0
g0 = base["guest_satisfaction_overall"].fillna(-1) == 0
print("  star==0:", int(s0.sum()), "| satisf==0:", int(g0.sum()),
      "| coincidem:", bool((s0 == g0).all()))
base["star_rating_clean"] = np.where(r0 & s0, np.nan, base["star_rating"])
base["guest_satisfaction_clean"] = np.where(r0 & g0, np.nan, base["guest_satisfaction_overall"])

base["number_of_reviews_host"] = pd.to_numeric(base["number_of_reviews_host"], errors="coerce")
base["star_rating_host"] = pd.to_numeric(base["star_rating_host"], errors="coerce")
hr0 = base["number_of_reviews_host"].fillna(-1) == 0
hs0 = base["star_rating_host"].fillna(-1) == 0
print("  host reviews==0:", int(hr0.sum()), "| host star==0:", int(hs0.sum()),
      "| coincidem:", bool(np.all(hr0 == hs0)))
base["star_rating_host_clean"] = np.where(hr0 & hs0, np.nan, base["star_rating_host"])

# ============================================================
# 2. AMENITIES PARSING
# ============================================================
print("=" * 78)
print("2. AMENITIES PARSING")
amen_vals = [parse_amen(s) for s in base["amenities"]]
n_total = len(amen_vals)
n_missing = sum(1 for v in amen_vals if v is None)
n_fail = sum(1 for v in amen_vals if v is not None and v < 0)
n_ok = sum(1 for v in amen_vals if v is not None and v >= 0)
print(f"  N total={n_total} | parseados={n_ok} | missing={n_missing} | falhas={n_fail}")
base["amenities_count"] = pd.Series([0 if v is None else max(v, 0) for v in amen_vals], index=base.index)

# ============================================================
# 3. FEATURES
# ============================================================
print("=" * 78)
print("3. FEATURES")
base["n_beds"] = pd.to_numeric(base["number_of_beds"], errors="coerce")
base["n_guests"] = pd.to_numeric(base["number_of_guests"], errors="coerce")
base["n_bathrooms"] = pd.to_numeric(base["number_of_bathrooms"], errors="coerce")
base["n_bedrooms"] = pd.to_numeric(base["number_of_bedrooms"], errors="coerce")
base["listing_type_low"] = base["listing_type"].astype(str).str.strip().str.lower()
base["cleaning_fee"] = pd.to_numeric(base["cleaning_fee"], errors="coerce")
for c in ["can_instant_book", "is_professional", "is_new_listing", "is_guest_favorite"]:
    base[c + "_b"] = base[c].apply(to_bool)
base["picture_count"] = pd.to_numeric(base["picture_count"], errors="coerce")
base["n_reviews"] = pd.to_numeric(base["number_of_reviews"], errors="coerce")
base["n_reviews_host"] = pd.to_numeric(base["number_of_reviews_host"], errors="coerce")
base["star_clean"] = base["star_rating_clean"]
base["guest_sat_clean"] = base["guest_satisfaction_clean"]
base["is_superhost_b"] = base["is_superhost"].apply(to_bool)
base["is_verified_b"] = base["is_verified"].apply(to_bool)
base["star_host_clean"] = base["star_rating_host_clean"]
base["years_host"] = pd.to_numeric(base["years_host"], errors="coerce")
base["months_host"] = pd.to_numeric(base["months_host"], errors="coerce")
base["suburb_clean"] = base["suburb"]
base["description_length"] = base["ad_description"].fillna("").astype(str).str.len()

NUMERIC_FEATS = ["n_beds", "n_guests", "n_bathrooms", "n_bedrooms", "cleaning_fee",
                 "picture_count", "n_reviews", "star_clean", "guest_sat_clean",
                 "n_reviews_host", "star_host_clean", "years_host", "months_host",
                 "amenities_count", "description_length"]
BOOL_FEATS = ["can_instant_book_b", "is_professional_b", "is_new_listing_b",
              "is_guest_favorite_b", "is_superhost_b", "is_verified_b"]
CAT_FEATS = ["suburb_clean", "listing_type_low"]
ALL_FEATS = NUMERIC_FEATS + BOOL_FEATS + CAT_FEATS

# ============================================================
# 3b. FEATURE AUDIT ampliado
# ============================================================
print("=" * 78)
print("3b. FEATURE AUDIT ampliado")
qa_rows = []
for f in ALL_FEATS:
    s = base[f]
    rec = {"feature": f, "N": int(s.notna().sum()), "missing_rate": round(s.isna().mean(), 4),
           "min": np.nan, "P01": np.nan, "P25": np.nan, "mediana": np.nan,
           "P75": np.nan, "P99": np.nan, "max": np.nan,
           "cardinalidade": int(s.nunique(dropna=False)), "variancia": None}
    if s.dtype.kind in "fi":
        num = s.dropna()
        if len(num):
            rec.update({"min": num.min(), "P01": num.quantile(0.01), "P25": num.quantile(0.25),
                        "mediana": num.median(), "P75": num.quantile(0.75), "P99": num.quantile(0.99),
                        "max": num.max(), "variancia": round(float(num.var()), 4) if len(num) > 1 else None})
    qa_rows.append(rec)
feature_qa = pd.DataFrame(qa_rows)
ext = []
for f in NUMERIC_FEATS:
    s = base[f]
    if s.notna().sum() < 10:
        continue
    q01, q99 = s.quantile(0.01), s.quantile(0.99)
    feature_qa.loc[feature_qa["feature"] == f, "extreme_for_review"] = int(((s < q01) | (s > q99)).sum())
save(feature_qa, "driver_feature_audit.csv")
drop_const = [f for f in ALL_FEATS if base[f].nunique(dropna=False) <= 1]
if drop_const:
    print("  constantes (fora do modelo, NÃO removidas da base):", drop_const)

# ============================================================
# 4. QUARTIS por target
# ============================================================
print("=" * 78)
print("4. QUARTIS por target")
quart_rows = []
for f in NUMERIC_FEATS:
    for tgt, tname in [("A_static_rev", "static revenue"), ("B_adr", "ADR"), ("C_unavail", "unavailability")]:
        d = base[[f, tgt]].replace([np.inf, -np.inf], np.nan).dropna()
        if d[f].nunique() < 4:
            continue
        try:
            d = d.assign(q=pd.qcut(d[f], 4, labels=False, duplicates="drop"))
        except Exception:
            continue
        for q, g in d.groupby("q"):
            quart_rows.append({"feature": f, "target": tname, "quartile": int(q),
                               "N": int(len(g)), "feature_min": g[f].min(), "feature_max": g[f].max(),
                               "target_median": g[tgt].median(),
                               "target_P25": g[tgt].quantile(0.25),
                               "target_P75": g[tgt].quantile(0.75)})
save(pd.DataFrame(quart_rows), "driver_numeric_quartiles.csv")

# ============================================================
# 6. Spearman + binário + categórico
# ============================================================
print("=" * 78)
print("6. ANÁLISE UNIVARIADA")
TARGETS = [("A_static_rev", "static revenue"), ("B_adr", "ADR"), ("C_unavail", "unavailability")]
num_rows = []
for f in NUMERIC_FEATS:
    for tgt, tname in TARGETS:
        dd = base[[f, tgt]].replace([np.inf, -np.inf], np.nan).dropna()
        rho = spearmanr(dd[f], dd[tgt]).correlation if len(dd) >= 10 else np.nan
        num_rows.append({"feature": f, "target": tname, "n_valid": len(dd), "spearman": rho})
save(pd.DataFrame(num_rows), "driver_numeric_associations.csv")

bin_rows = []
for f in BOOL_FEATS:
    for tgt, tname in TARGETS:
        d = base[[f, tgt]].dropna(subset=[tgt, f])
        if len(d[d[f] == 1]) < 1 or len(d[d[f] == 0]) < 1:
            continue
        tru, fls = d[d[f] == 1][tgt], d[d[f] == 0][tgt]
        bin_rows.append({"feature": f, "target": tname, "N_true": int(len(tru)),
                         "N_false": int(len(fls)), "med_true": tru.median(),
                         "med_false": fls.median(),
                         "diff_abs": tru.median() - fls.median(),
                         "diff_pct": ((tru.median() - fls.median()) / fls.median())
                         if fls.median() else np.nan})
save(pd.DataFrame(bin_rows), "driver_binary_associations.csv")

cat_rows = []
for f in CAT_FEATS:
    for tgt, tname in TARGETS:
        d = base[[f, tgt]].dropna(subset=[tgt])
        g = d.groupby(f, dropna=False)[tgt].agg(["median", "count",
                                                 lambda s: s.quantile(0.25),
                                                 lambda s: s.quantile(0.75)])
        g.columns = ["mediana", "N", "p25", "p75"]
        g = g.reset_index().rename(columns={f: "categoria"})
        g["target"] = tname
        g["faixa_amostral"] = g["N"].apply(faixa)
        cat_rows.append(g)
save(pd.concat(cat_rows), "driver_categorical_associations.csv")

# ============================================================
# 5/7. CV: GroupKFold (owner) principal + KFold sensibilidade; Dummy nas folds
# ============================================================
print("=" * 78)
print("7. CV: GroupKFold(owner) principal + Dummy(mesmas folds)")


def feats_for(add_suburb):
    num = [f for f in NUMERIC_FEATS if base[f].nunique() > 1]
    binf = [f for f in BOOL_FEATS if base[f].nunique() > 1]
    cat = (["suburb_clean"] if add_suburb else []) + ["listing_type_low"]
    return num, binf, cat


def make_design(target_col, add_suburb):
    num, binf, cat = feats_for(add_suburb)
    featz = num + binf + cat
    d = base[featz + [target_col, "owner_id"]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col]).copy()
    return d[featz], d[target_col], d["owner_id"].fillna("__missing__").values, num, binf, cat


def make_pipe(num, binf, cat):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num + binf),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre),
                     ("rf", RandomForestRegressor(n_estimators=150, random_state=RS, n_jobs=-1))])


cv_results = []
for tgt, tname in TARGETS:
    for sub in [True, False]:
        X, y, groups, num, binf, cat = make_design(tgt, sub)
        pipe = make_pipe(num, binf, cat)
        gk = GroupKFold(n_splits=5)
        mae_g = -cross_val_score(pipe, X, y, groups=groups, cv=gk, scoring="neg_mean_absolute_error", error_score="raise")
        r2_g = cross_val_score(pipe, X, y, groups=groups, cv=gk, scoring="r2", error_score="raise")
        mae_dum = -cross_val_score(DummyRegressor(strategy="median"), X, y, groups=groups, cv=gk,
                                   scoring="neg_mean_absolute_error")
        cv_results.append({"target": tname, "with_suburb": sub, "validation": "grouped_by_owner",
                           "n": len(y), "rf_mae_mean": mae_g.mean(), "rf_mae_std": mae_g.std(),
                           "rf_r2_mean": r2_g.mean(), "rf_r2_std": r2_g.std(),
                           "dummy_mae_mean": mae_dum.mean(), "dummy_mae_std": mae_dum.std(),
                           "rf_vs_dummy_ratio": mae_g.mean() / mae_dum.mean()})
        # sensibilidade KFold aleatório
        kf = KFold(n_splits=5, shuffle=True, random_state=RS)
        mae_k = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error", error_score="raise")
        r2_k = cross_val_score(pipe, X, y, cv=kf, scoring="r2", error_score="raise")
        mae_k_d = -cross_val_score(DummyRegressor(strategy="median"), X, y, cv=kf,
                                   scoring="neg_mean_absolute_error")
        cv_results.append({"target": tname, "with_suburb": sub, "validation": "random_listing_kfold",
                           "n": len(y), "rf_mae_mean": mae_k.mean(), "rf_mae_std": mae_k.std(),
                           "rf_r2_mean": r2_k.mean(), "rf_r2_std": r2_k.std(),
                           "dummy_mae_mean": mae_k_d.mean(), "dummy_mae_std": mae_k_d.std(),
                           "rf_vs_dummy_ratio": mae_k.mean() / mae_k_d.mean()})
save(pd.DataFrame(cv_results), "driver_model_cv.csv")
for r in cv_results:
    print(f"  [{r['validation']}] {r['target']} sub={r['with_suburb']}: "
          f"RF_MAE={r['rf_mae_mean']:.1f} R2={r['rf_r2_mean']:.3f} "
          f"Dummy={r['dummy_mae_mean']:.1f} RF/Dummy={r['rf_vs_dummy_ratio']:.2f}")

# ============================================================
# 8/9. PERMUTATION IMPORTANCE (Pipeline, espaço original, split por host)
# ============================================================
print("=" * 78)
print("8/9. PERMUTATION IMPORTANCE (split por host, Pipeline no espaço original)")
perm_rows = []
for tgt, tname in TARGETS:
    for sub in [True, False]:
        X, y, groups, num, binf, cat = make_design(tgt, sub)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RS)
        (tr, te), = gss.split(X, y, groups)
        # checar interseção de OWNERS treino/teste
        own_tr = set(groups[tr]); own_te = set(groups[te])
        inter = own_tr & own_te
        assert len(inter) == 0, f"owner crossover! n={len(inter)}"
        pipe = make_pipe(num, binf, cat)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pipe_score = pipe.score(X.iloc[te], y.iloc[te])
        te_shape = (y.iloc[te] - pipe.predict(X.iloc[te])).abs().mean()
        pred = DummyRegressor(strategy="median").fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        dum_mae = (y.iloc[te] - pred).abs().mean()
        perm = permutation_importance(pipe, X.iloc[te], y.iloc[te], n_repeats=10,
                                       random_state=RS, scoring="neg_mean_absolute_error")
        for i, f in enumerate(X.columns):
            perm_rows.append({"target": tname, "with_suburb": sub, "feature": f,
                              "perm_imp": perm.importances_mean[i],
                              "holdout_n": int(len(te))})
save(pd.DataFrame(perm_rows), "driver_permutation_importance.csv")
print("  permutation importance salva (exploratória; ver holdout vs dummy).")

# ============================================================
# 11/12. CENTRO 1Q vs 2Q (corrigir direção) + superhost reconcile
# ============================================================
print("=" * 78)
print("11. CENTRO 1Q vs 2Q (direção correta)")
base["bedr"] = base["n_bedrooms"]
base["typ_low"] = base["listing_type_low"]
c1 = base[(base["typ_low"] == "apartamento") & (base["bedr"] == 1) & (base["suburb_clean"] == "Centro")]
c2 = base[(base["typ_low"] == "apartamento") & (base["bedr"] == 2) & (base["suburb_clean"] == "Centro")]
for nm, gg in [("1Q", c1), ("2Q", c2)]:
    gg = gg[gg["A_static_rev"].notna()]
    if len(gg) == 0:
        continue
    sh_rate = gg["is_superhost_b"].mean()
    print(f"  {nm}: N={len(gg)} static={gg['A_static_rev'].median():.0f} "
          f"ADR={gg['B_adr'].median():.0f} unav={gg['C_unavail'].median():.3f} "
          f"superhost_rate={sh_rate:.3f}")

# ============================================================
# 13. TABELA 4 CANDIDATOS (02/03 integrado)
# ============================================================
print("=" * 78)
print("13. TABELA 4 CANDIDATOS (integrando 02/03)")
cand4 = [("Morretes", 2, "apartamento"), ("Meia Praia", 2, "apartamento"),
         ("Centro", 2, "apartamento"), ("Centro", 1, "apartamento")]
cand_rows = []
inv_idx = inv.set_index(["listing_type", "bedrooms", "suburb"])
for (subb, bed, tp) in cand4:
    key = (tp, bed, subb)
    if key not in inv_idx.index:
        continue
    invr = inv_idx.loc[key]
    # estrutura / anúncio promovi de base (por segmento com preço)
    m = base[(base["typ_low"] == tp) & (base["bedr"] == bed) & (base["suburb_clean"] == subb)]
    mp = m[m["A_static_rev"].notna()]
    rows = {
        "perfil": f"{tp} {bed}Q {subb}",
        "N_airbnb": int(invr["N_airbnb_91"]), "N_sale": int(invr["N_sale"]),
        "N_pickup": int(invr["N_pickup"]),
        "ADR": round(invr["ADR_91_median"], 1), "unavailability": round(invr["unavailability_91_median"], 3),
        "static_rev_proxy": round(invr["static_rev_91_median"], 1),
        "median_sale_price": invr["median_sale_price"],
        "capital_efficiency_proxy": round(invr["capital_efficiency_proxy_91"], 5),
        "net_pickup_value_median": round(invr["net_pickup_value_median"], 1),
        "net_transition_rate": round(invr["net_transition_rate_median"], 4),
        "evidence_tier": invr["evidence_tier"],
        "guests": mp["n_guests"].median(), "beds": mp["n_beds"].median(),
        "bathrooms": mp["n_bathrooms"].median(),
        "cleaning_fee": mp["cleaning_fee"].median(),
        "reviews": mp["n_reviews"].median(), "star_clean": mp["star_clean"].median(),
        "guest_sat_clean": mp["guest_sat_clean"].median(),
        "guest_fav_rate": round(mp["is_guest_favorite_b"].mean(), 3),
        "professional_rate": round(mp["is_professional_b"].mean(), 3),
        "superhost_rate": round(mp["is_superhost_b"].mean(), 3),
        "amenities_count": mp["amenities_count"].median(),
        "picture_count": mp["picture_count"].median(),
    }
    cand_rows.append({
        "perfil": f"{tp} {bed}Q {subb}",
        "N_airbnb": int(invr["N_airbnb_91"]), "N_sale": int(invr["N_sale"]),
        "N_pickup": int(invr["N_pickup"]),
        "ADR": round(float(invr["ADR_91_median"]), 1),
        "unavailability": round(float(invr["unavailability_91_median"]), 3),
        "static_rev_proxy": round(float(invr["static_rev_91_median"]), 1),
        "median_sale_price": float(invr["median_sale_price"]),
        "capital_efficiency_proxy": round(float(invr["capital_efficiency_proxy_91"]), 5),
        "net_pickup_value_median": round(float(invr["net_pickup_value_median"]), 1),
        "net_transition_rate": round(float(invr["net_transition_rate_median"]), 4),
        "evidence_tier": invr["evidence_tier"],
        "guests": mp["n_guests"].median(), "beds": mp["n_beds"].median(),
        "bathrooms": mp["n_bathrooms"].median(),
        "cleaning_fee": mp["cleaning_fee"].median(),
        "reviews": mp["n_reviews"].median(), "star_clean": mp["star_clean"].median(),
        "guest_sat_clean": mp["guest_sat_clean"].median(),
        "guest_fav_rate": round(float(mp["is_guest_favorite_b"].mean()), 3),
        "professional_rate": round(float(mp["is_professional_b"].mean()), 3),
        "superhost_rate": round(float(mp["is_superhost_b"].mean()), 3),
        "amenities_count": mp["amenities_count"].median(),
        "picture_count": mp["picture_count"].median(),
    })
cand_df = pd.DataFrame(cand_rows)
save(cand_df, "candidate_driver_comparison.csv")
print("   candidatos salvos em candidate_driver_comparison.csv")

# ============================================================
# 14. FRONTEIRA DE PARETO (apenas os 4 candidatos)
# ============================================================
print("=" * 78)
print("14. FRONTEIRA DE PARETO (max cap_eff x max pickup)")
pts = cand_df[["perfil", "capital_efficiency_proxy", "net_pickup_value_median"]].copy()
points = pts.to_dict("records")
ndv = []
for i, a in enumerate(points):
    dominated = False
    for j, b in enumerate(points):
        if i == j:
            continue
        if (b["capital_efficiency_proxy"] >= a["capital_efficiency_proxy"] and
            b["net_pickup_value_median"] >= a["net_pickup_value_median"] and
            (b["capital_efficiency_proxy"] > a["capital_efficiency_proxy"] or
             b["net_pickup_value_median"] > a["net_pickup_value_median"])):
            dominated = True
            break
    ndv.append("ND" if not dominated else "DOM")
pts["nao_dominado"] = ndv
save(pts, "investment_pareto_front.csv")
for _, r in pts.iterrows():
    print(f"   [{r['nao_dominado']}] {r['perfil']}: cap_eff={r['capital_efficiency_proxy']:.4f} "
          f"pickup={r['net_pickup_value_median']:.0f}")

# ============================================================
# 15. FIGURAS
# ============================================================
print("=" * 78)
print("15. FIGURAS")
try:
    plt.figure(figsize=(7, 5))
    for _, r in pts.iterrows():
        col = "#2ca02c" if r["nao_dominado"] == "ND" else "#d62728"
        plt.scatter(r["capital_efficiency_proxy"], r["net_pickup_value_median"],
                    c=col, s=90)
        plt.annotate(r["perfil"], (r["capital_efficiency_proxy"], r["net_pickup_value_median"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8)
    plt.xlabel("capital_efficiency_proxy_91"); plt.ylabel("net_pickup_value_median")
    plt.title("Pareto 4 candidatos (verde=ND, vermelho=DOM)")
    plt.tight_layout(); plt.savefig(FIG / "candidate_pareto.png", dpi=110); plt.close()
    print("   fig: candidate_pareto.png")
except Exception as e:
    print("  aviso pareto fig:", e)
try:
    comp = cand_df.set_index("perfil")[["static_rev_proxy", "capital_efficiency_proxy",
                                        "net_pickup_value_median", "unavailability"]].copy()
    comp_norm = comp.div(comp.max(), axis=1)
    ax2 = comp_norm.plot.bar(figsize=(9, 5), rot=45)
    ax2.set_title("Comparação dos 4 candidatos (normalizado 0-1; sem score)")
    ax2.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "candidate_comparison.png", dpi=110); plt.close()
    print("   fig: candidate_comparison.png")
except Exception as e:
    print("  aviso fig candidate_comparison:", e)

print("\nDONE")