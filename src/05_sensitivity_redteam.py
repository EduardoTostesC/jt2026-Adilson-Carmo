"""
Etapa 05 — SENSITIVITY & RED-TEAM (PATCH FINAL).

Replicações desta rodada:
  - Seleção VivaReal via canônico 03 (map vivareal_suburb_mapping.csv);
  - assertions N/mediana canônicos antes do bootstrap;
  - bootstrap principal: owner-cluster Airbnb + listing-bootstrap VivaReal canônico;
  - auditoria + sensibilidade advertiser-cluster;
  - zero-rate de condomínio/IPTU entre observados;
  - cobertura dupla (any Price_AV vs snapshot 20/01 H91);
  - intervalos em %; cap_eff entre 0 e 1;
  - red-team com value_counts.

Não altera 01-04; não cria nova metodologia; não faz recomendação final.
"""

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "tables"
FIG = Path(__file__).resolve().parent.parent / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
RS = 42
rng = np.random.default_rng(RS)
NREP = 2000

CANDIDATES = [
    ("Morretes", 2, "apartamento"),
    ("Meia Praia", 2, "apartamento"),
    ("Centro", 2, "apartamento"),
    ("Centro", 1, "apartamento"),
]


def lab(seg):
    return f"{seg[2]} {seg[1]}Q {seg[0]}"


def save(df, name):
    df.to_csv(OUT / name, index=False)
    print("  saved ->", name, "(", len(df), "linhas)")


def pct(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {"median": np.nan, "P2_5": np.nan, "P25": np.nan, "P75": np.nan, "P97_5": np.nan}
    return {"median": float(np.median(vals)), "P2_5": float(np.percentile(vals, 2.5)),
            "P25": float(np.percentile(vals, 25)), "P75": float(np.percentile(vals, 75)),
            "P97_5": float(np.percentile(vals, 97.5))}


# ============ CANON BAIERO (igual 03) ============
def load_submap():
    m = pd.read_csv(OUT / "vivareal_suburb_mapping.csv")
    return dict(zip(m["suburb_original"].astype(str).str.strip(), m["suburb_canonical"]))


submap = load_submap()


def canon_suburb(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    return submap.get(str(s).strip())


# ============ CARGA ============
print("=" * 73)
print("CARGA")
lm = pd.read_csv(OUT / "listing_metrics.csv", dtype={"airbnb_listing_id": str})
pk = pd.read_csv(OUT / "pickup_metrics.csv", dtype={"airbnb_listing_id": str})
inv = pd.read_csv(OUT / "investment_segments.csv")
details = pd.read_csv(DATA / "Details_Itapema.csv", dtype={"airbnb_listing_id": str, "owner_id": str}, low_memory=False)
mesh = pd.read_csv(DATA / "Mesh_Ids_Data_Itapema.csv", dtype={"airbnb_listing_id": str})
vr = pd.read_csv(DATA / "VivaReal_Itapema.csv", dtype={"listing_id": str}, low_memory=False)
vr = vr.drop_duplicates("listing_id", keep="first").copy()
for c in ["sale_price", "usable_area", "monthly_condo_fee", "yearly_iptu"]:
    vr[c] = pd.to_numeric(vr[c], errors="coerce")
vr["suburb_original"] = vr["suburb"]
vr["suburb_canonical"] = vr["suburb"].apply(canon_suburb)

owner_map = details.set_index("airbnb_listing_id")["owner_id"].to_dict()
suburb_map = mesh.set_index("airbnb_listing_id")["suburb"].to_dict()

lm = lm.copy()
lm["owner_id"] = lm["airbnb_listing_id"].map(owner_map)
lm["bedo"] = pd.to_numeric(lm["number_of_bedrooms"], errors="coerce")
lm["typ"] = lm["listing_type"].astype(str).str.strip().str.lower()
pl = pk.copy()
pl["owner_id"] = pl["airbnb_listing_id"].map(owner_map)
inv_idx = inv.set_index(["listing_type", "bedrooms", "suburb"])


def seg_listings(seg):
    subb, b, tp = seg
    return lm[(lm["suburb"] == subb) & (lm["bedo"] == b) & (lm["typ"] == tp)]


def seg_pickup(seg):
    ids = set(seg_listings(seg)["airbnb_listing_id"])
    return pl[pl["airbnb_listing_id"].isin(ids)]


def seg_viv(seg):
    subb, b, tp = seg
    return vr[(vr["suburb_canonical"] == subb) & (vr["bedrooms"] == b) & (vr["listing_type"] == tp)]


def static_prices(seg):
    m = seg_listings(seg)
    m = m[m["n_prices_observed_H_91"].fillna(0) > 0]
    return m[["airbnb_listing_id", "owner_id", "static_revenue_proxy_H_91"]].dropna(
        subset=["static_revenue_proxy_H_91"])


# ============ 2. ASSERTIONS N e mediana VivoReal canônico ============
print("=" * 73)
print("2. ASSERTIONS Ns/medianas VivaReal (canonical, =03)")
EXP = {("Centro", 1): (22, 890000), ("Centro", 2): (89, 1150000),
       ("Meia Praia", 2): (243, 1080000), ("Morretes", 2): (1037, 790000)}
for (subb, b, tp) in CANDIDATES:
    sp = seg_viv((subb, b, tp))["sale_price"].dropna()
    expN, expM = EXP[(subb, b)]
    assert len(sp) == expN, f"N divergente {lab((subb,b,tp))}: {len(sp)}!={expN}"
    assert abs(float(np.median(sp)) - expM) < 1, f"mediana divergente {lab((subb,b,tp))}"
    ir = inv_idx.loc[(tp, b, subb)]
    assert int(ir["N_sale"]) == int(len(sp)), f"N_sale {lab((subb,b,tp))} != investment_segments"
print("  ASSERTIONS OK (N e mediana=03; igual a investment_segments)")

# ============================================================
# 3. BOOTSTRAP CAPITAL EFFICIENCY (principal + advertiser sensibilidade)
# ============================================================
print("=" * 73)
print("3. BOOTSTRAP CAPITAL EFFICIENCY")


def owner_static_draw_main(m):
    """Um draw: cluster owner/bootstrap do static_revenue_proxy H91."""
    owners = list(m["owner_id"].dropna().unique())
    if not owners:
        return np.nan
    oid = {o: m.index[m["owner_id"] == o].tolist() for o in owners}
    so = rng.choice(owners, size=len(owners), replace=True)
    rows = [i for o in so for i in oid[o]]
    vals = m.loc[rows, "static_revenue_proxy_H_91"].dropna().values
    return float(np.median(vals)) if len(vals) else np.nan


def listing_sale_draw(sp):
    sp = np.asarray(sp, dtype=float)
    if len(sp) == 0:
        return np.nan
    return float(np.median(rng.choice(sp, size=len(sp), replace=True)))


def adv_sale_draw(v):
    """(advertiser-cluster) draw da mediana de sale_price."""
    adv = v["advertiser_name"].dropna()
    if len(adv) == 0:
        return np.nan
    adv2 = {a: v.index[v["advertiser_name"] == a].tolist() for a in adv.unique()}
    advs = list(adv.unique())
    so = rng.choice(advs, size=len(advs), replace=True)
    rows = [i for a in so for i in adv2[a]]
    vals = v.loc[rows, "sale_price"].dropna().values
    return float(np.median(vals)) if len(vals) else np.nan


# ============ MATRIZ H30/H60/H91 (para ranking/figuras) ============
hor_rows = []
for seg in CANDIDATES:
    invr = inv_idx.loc[(seg[2], seg[1], seg[0])]
    med_sale = invr["median_sale_price"]
    d = seg_listings(seg)
    for H in [30, 60, 91]:
        scol = f"static_revenue_proxy_H_{H}"
        pcol = f"n_prices_observed_H_{H}"
        wp = d[d[pcol].fillna(0) > 0]
        s = wp[scol].dropna()
        med = s.median() if len(s) else np.nan
        cap = med / med_sale if pd.notna(med) and med_sale else np.nan
        hor_rows.append({"candidato": lab(seg), "horizonte": f"H{H}",
                         "N_airbnb_snapshot": len(d), "N_com_proxy": int(len(s)),
                         "median_static_proxy": med,
                         "P25": s.quantile(0.25) if len(s) else np.nan,
                         "P75": s.quantile(0.75) if len(s) else np.nan,
                         "median_sale_price": med_sale, "capital_efficiency": cap})
hor_df = pd.DataFrame(hor_rows)
save(hor_df, "candidate_horizon_sensitivity.csv")

boot_rows = []
boot_adv_rows = []
for seg in CANDIDATES:
    mprice = static_prices(seg)
    v = seg_viv(seg)
    sp = v["sale_price"].dropna().values
    owners = int(mprice["owner_id"].nunique())
    # principal: owner-cluster Airbnb + listing bootstrap Viva
    draws = np.array([owner_static_draw_main(mprice) / listing_sale_draw(sp) for _ in range(NREP)])
    draws = draws[~np.isnan(draws)]
    # sensibilidade: owner-cluster Airbnb + advertiser-cluster Viva
    adv_feasible = (v["advertiser_name"].notna().mean() >= 0.8) and (v["advertiser_name"].dropna().nunique() >= 5)
    if adv_feasible:
        adv_draws = np.array([owner_static_draw_main(mprice) / adv_sale_draw(v) for _ in range(NREP)])
        adv_draws = adv_draws[~np.isnan(adv_draws)]
        st_adv = pct(adv_draws)
    st_main = pct(draws)
    rec = {"candidato": lab(seg), "N_airbnb": int(len(mprice)), "N_owners": owners,
           "N_sale": int(len(sp)),
           **{f"cluster_{k}": v for k, v in st_main.items()},
           "metodo": "owner_cluster_airbnb+listing_bootstrap_vivo"}
    boot_rows.append(rec)
    if adv_feasible:
        boot_adv_rows.append({"candidato": lab(seg), "advertiser_feasible": True,
                              **{f"adv_{k}": v for k, v in st_adv.items()}})
    else:
        boot_adv_rows.append({"candidato": lab(seg), "advertiser_feasible": False})

boot_df = pd.DataFrame(boot_rows)
save(boot_df, "bootstrap_efficiency.csv")
print(boot_df[["candidato", "N_airbnb", "N_owners", "N_sale",
               "cluster_median", "cluster_P2_5", "cluster_P97_5"]].to_string(index=False))
boot_adv_df = pd.DataFrame(boot_adv_rows)
save(boot_adv_df, "bootstrap_efficiency_advertiser_sensitivity.csv")

# ============================================================
# 4. ADVERTISER AUDIT
# ============================================================
print("=" * 73)
print("4. ADVERTIER AUDIT")
adv_aud = []
for seg in CANDIDATES:
    v = seg_viv(seg)
    adv = v["advertiser_name"].dropna()
    miss = v["advertiser_name"].isna().mean()
    n_uniq = adv.nunique()
    largest = adv.value_counts().iloc[0] if len(adv) else 0
    top5 = adv.value_counts().head(5).sum() if len(adv) else 0
    adv_aud.append({"candidato": lab(seg), "N_sale": len(v),
                    "N_advertiser_nomissing": int(len(adv)),
                    "missing_rate_advertiser": round(miss, 4),
                    "N_advertisers_unicos": n_uniq,
                    "largest_advertiser_share": round(largest / len(adv), 4) if len(adv) else np.nan,
                    "top5_advertisers_share": round(top5 / len(adv), 4) if len(adv) else np.nan})
save(pd.DataFrame(adv_aud), "vivareal_advertiser_concentration.csv")
print(pd.DataFrame(adv_aud).to_string(index=False))

# ============================================================
# 6. BOOTSTRAP PAIRWISE WIN RATE (principal + advertiser sens)
# ============================================================
print("=" * 73)
print("6. PAIRWISE WIN RATE")


def build_draws(method):
    out = {}
    for seg in CANDIDATES:
        mprice = static_prices(seg)
        v = seg_viv(seg)
        sp = v["sale_price"].dropna().values
        if method == "listing":
            arr = np.array([owner_static_draw_main(mprice) / listing_sale_draw(sp) for _ in range(NREP)])
        else:
            arr = np.array([owner_static_draw_main(mprice) / adv_sale_draw(v) for _ in range(NREP)])
        out[seg] = arr[~np.isnan(arr)]
    return out


def build_pairwise(d_map, outname):
    pw = []
    for a, b in combinations(CANDIDATES, 2):
        da, db = d_map[a], d_map[b]
        n = min(len(da), len(db))
        pw.append({"A": lab(a), "B": lab(b),
                   "win_rate_A_over_B": round(float(np.mean(da[:n] > db[:n])), 4), "n": int(n)})
    save(pd.DataFrame(pw), outname)
    return pd.DataFrame(pw)


draws_listing = build_draws("listing")
pw_main = build_pairwise(draws_listing, "bootstrap_pairwise.csv")
print(pw_main.to_string(index=False))
# advertiser sensibilidade
try:
    draws_adv = build_draws("advertiser")
    pw_adv = build_pairwise(draws_adv, "bootstrap_pairwise_advertiser_sensitivity.csv")
    print(pw_adv.to_string(index=False))
except Exception as e:
    print("  advertiser pairwise não gerado:", e)

# ============================================================
# 7. PICKUP / MOMENTUM
# ============================================================
print("=" * 73)
print("7. PICKUP / MOMENTUM")
pick_rows = []
for seg in CANDIDATES:
    p = seg_pickup(seg)
    nv = p["net_pickup_value_proxy"].dropna()
    tr = p["net_transition_rate"].dropna()
    rec = {"candidato": lab(seg), "N_pickup": len(p),
           "available_to_unavailable_elev": p["available_to_unavailable_nights"].median(),
           "unavailable_to_available_median": p["unavailable_to_available_nights"].median(),
           "net_pickup_nights_median": p["net_unavailability_pickup"].median(),
           "net_pickup_value_median": nv.median(), "net_pickup_value_mean": nv.mean(),
           "net_pickup_value_p25": nv.quantile(0.25), "net_pickup_value_p75": nv.quantile(0.75),
           "net_transition_rate_median": tr.median(), "net_transition_rate_p25": tr.quantile(0.25),
           "net_transition_rate_p75": tr.quantile(0.75)}
    pick_rows.append(rec)
pickup_df = pd.DataFrame(pick_rows)
save(pickup_df, "pickup_sensitivity.csv")
print(pickup_df[["candidato", "N_pickup", "net_pickup_value_median",
               "net_transition_rate_median"]].to_string(index=False))

# ============================================================
# 8/9. CUSTOS + COVERAGE
# ============================================================
print("=" * 73)
print("8. CUSTOS OBSERVÁVEIS (zero-rate corrigido) + 9. coverage")


def rate_report(series):
    obs = series.dropna()
    return {"N_observed": int(len(obs)),
            "missing_rate": round(series.isna().mean(), 4),
            "N_zero_observed": int((obs == 0).sum()),
            "zero_rate_among_observed": round((obs == 0).mean(), 4) if len(obs) else np.nan}


cost_rows = []
for seg in CANDIDATES:
    v = seg_viv(seg)
    ir = inv_idx.loc[(seg[2], seg[1], seg[0])]
    cf = rate_report(pd.to_numeric(v["monthly_condo_fee"], errors="coerce"))
    ip = rate_report(pd.to_numeric(v["yearly_iptu"], errors="coerce"))
    cc = ir["median_monthly_condo_fee"]
    yt = ir["median_yearly_iptu"]
    p91 = ir["static_rev_91_median"]
    med_sale = ir["median_sale_price"]
    proxy_cost = p91 - 3 * cc - (91 / 365) * yt
    cost_rows.append({"candidato": lab(seg), **{f"condo_{k}": val for k, val in cf.items()},
                      **{f"iptu_{k}": val for k, val in ip.items()},
                      "partial_observed_cost_proxy_91": proxy_cost,
                      "partial_cost_capital_efficiency": proxy_cost / med_sale if med_sale else np.nan})
cost_df = pd.DataFrame(cost_rows)
save(cost_df, "partial_cost_sensitivity.csv")

# coverage dupla
print("=" * 73)
print("9. COVERAGE (any Price_AV vs snapshot 20/01 H91)")
cov_rows = []
price_any_ids = set(pd.read_csv(DATA / "Price_AV_Itapema.csv", dtype={"airbnb_listing_id": str})["airbnb_listing_id"])
for seg in CANDIDATES:
    subb, b, tp = seg
    dd = details.copy()
    dd["bed0"] = pd.to_numeric(dd["number_of_bedrooms"], errors="coerce")
    dd["typ0"] = dd["listing_type"].astype(str).str.strip().str.lower()
    dd["sub0"] = dd["airbnb_listing_id"].map(suburb_map)
    n_det = int(((dd["typ0"] == tp) & (dd["bed0"] == b) & (dd["sub0"] == subb)).sum())
    mseg = seg_listings(seg)
    ids_all = set(dd.loc[(dd["typ0"] == tp) & (dd["bed0"] == b) & (dd["sub0"] == subb), "airbnb_listing_id"])
    n_any = int(len(ids_all & price_any_ids))
    # snapshot = listings do segmento com preço observado em 20/01 (H91)
    n_snap = int((mseg["n_prices_observed_H_91"].fillna(0) > 0).sum())
    cov_rows.append({"cand": lab(seg), "N_details": n_det,
                     "N_any_price_av": n_any,
                     "coverage_any_price_av": n_any / n_det if n_det else np.nan,
                     "N_snapshot_20jan_H91": n_snap,
                     "coverage_snapshot_20jan_H91": n_snap / n_det if n_det else np.nan})
cov_df = pd.DataFrame(cov_rows)
save(cov_df, "coverage_bias_candidates.csv")
print(cov_df.to_string(index=False))

# ============================================================
# 18. RED-TEAM
# ============================================================
print("=" * 73)
print("10. RED-TEAM 10 ataques")
risks = [
    ("semântica Price_AV", "crítico"), ("indisponível != reserva", "crítico"),
    ("selection bias", "crítico"), ("custos não observados", "crítico"),
    ("sazonalidade", "importante"), ("asking price VivaReal", "importante"),
    ("dependência por host", "importante"), ("tamanho amostral", "importante"),
    ("construção static proxy", "importante"), ("Airbnb != VivaReal imóvel", "secundário"),
]
red = pd.DataFrame(risks, columns=["risco", "severidade"])
red["id"] = np.arange(1, len(red) + 1)
red["evidência_sustenta_risco"] = "diversos"
red["impacto"] = "proxy pode mudar"
red["mitigação"] = "rotular proxy; manter N explícito"
red["permanece_aberto"] = "sim"
red.to_csv(OUT / "redteam_risks.csv", index=False)
print("  contagem severidade (calculada):")
print(red["severidade"].value_counts().to_string())
assert len(red) == 10, "devem ser exatamente 10 riscos"

# ============================================================
# 10. ASSERT escala cap_eff (0,1) + INTERVALOS em %
# ============================================================
print("=" * 73)
print("CAP EFF — asserts de escala e intervalos em %")
boot_df = pd.read_csv(OUT / "bootstrap_efficiency.csv")
for seg in CANDIDATES:
    r = boot_df[boot_df["candidato"] == lab(seg)].iloc[0]
    for col in ["cluster_median", "cluster_P2_5", "cluster_P25", "cluster_P75", "cluster_P97_5"]:
        val = r[col]
        assert not np.isnan(val), f"{lab(seg)} {col} NaN"
        assert 0 < val < 1, f"{lab(seg)} {col} fora de (0,1): {val}"
    print(f"  {lab(seg)}: {r['cluster_median']*100:.2f}% "
          f"[{r['cluster_P2_5']*100:.2f}% – {r['cluster_P97_5']*100:.2f}%]")

# ============================================================
# 19/20. DECISION EVIDENCE (sem score) + FIGURAS
# ============================================================
print("=" * 73)
print("19/20. DECISION EVIDENCE + FIGURAS")
final_rows = []
for seg in CANDIDATES:
    ir = inv_idx.loc[(seg[2], seg[1], seg[0])]
    m = seg_listings(seg)
    boot_r = boot_df[boot_df["candidato"] == lab(seg)].iloc[0]
    pick_r = pickup_df[pickup_df["candidato"] == lab(seg)].iloc[0]
    cost_r = cost_df[cost_df["candidato"] == lab(seg)].iloc[0]
    ranks = {}
    for H in [30, 60, 91]:
        hsub = hor_df[hor_df["horizonte"] == f"H{H}"].dropna(subset=["capital_efficiency"]).sort_values(
            "capital_efficiency", ascending=False).reset_index(drop=True)
        hits = hsub.index[hsub["candidato"] == lab(seg)].tolist()
        ranks[H] = int(hits[0] + 1) if hits else np.nan
    final_rows.append({
        "candidato": lab(seg),
        "N_airbnb": int(ir["N_airbnb_91"]),
        "N_owner_airbnb": int(static_prices(seg)["owner_id"].nunique()),
        "N_sale": int(ir["N_sale"]), "N_pickup": int(ir["N_pickup"]),
        "efficiency_H30": float(ir["capital_efficiency_proxy_30"]), "rank_H30": ranks[30],
        "efficiency_H60": float(ir["capital_efficiency_proxy_60"]), "rank_H60": ranks[60],
        "efficiency_H91": float(ir["capital_efficiency_proxy_91"]), "rank_H91": ranks[91],
        "conservative_91": ir["conservative_91"], "base_91": ir["base_91"], "optimistic_91": ir["optimistic_91"],
        "bootstrap_median": boot_r["cluster_median"], "bootstrap_P2_5": boot_r["cluster_P2_5"],
        "bootstrap_P97_5": boot_r["cluster_P97_5"],
        "pickup_median": pick_r["net_pickup_value_median"],
        "net_transition_rate": pick_r["net_transition_rate_median"],
        "partial_cost_eff": cost_r["partial_cost_capital_efficiency"],
        "evidence_tier": ir["evidence_tier"],
    })
final_df = pd.DataFrame(final_rows)
# pareto_status na base investment_segments: não-dominado por cap_eff91 x pickup
cand96 = [(lab(s), s) for s in CANDIDATES]
pts = [{"cand": lab(s),
        "cap": float(inv_idx.loc[(s[2], s[1], s[0])]["capital_efficiency_proxy_91"]),
        "pk": float(inv_idx.loc[(s[2], s[1], s[0])]["net_pickup_value_median"])} for s in CANDIDATES]
stat = {}
for i, a in enumerate(pts):
    dom = False
    for j, b in enumerate(pts):
        if i == j:
            continue
        if (b["cap"] >= a["cap"] and b["pk"] >= a["pk"] and
                (b["cap"] > a["cap"] or b["pk"] > a["pk"])):
            dom = True
            break
    stat[a["cand"]] = "ND" if not dom else "DOM"
final_df["pareto_status"] = final_df["candidato"].map(stat)
# assertion N_owner consistente com bootstrap (owners dos listings H91)
boot_df = pd.DataFrame(boot_rows)
boot_owners = boot_df.set_index("candidato")["N_owners"].to_dict()
for _, r in final_df.iterrows():
    assert int(r["N_owner_airbnb"]) == int(boot_owners[r["candidato"]]), \
        f"N_owner inconsistente {r['candidato']}"
print("  assertion N_owner (uso H91) OK.")
save(final_df, "final_decision_evidence.csv")
print("  final_decision_evidence.csv atualizado (sem peso/nota).")

# ============================================================
# FIGURAS
# ============================================================
print("=" * 73)
print("FIGURAS")
# 1) sensitivity_eff var (H30/60/91)
try:
    plt.figure(figsize=(8, 5))
    for seg in CANDIDATES:
        vals = [hor_df[(hor_df["horizonte"] == f"H{h}") & (hor_df["candidato"] == lab(seg))][
            "capital_efficiency"].iloc[0] for h in [30, 60, 91]]
        plt.plot([30, 60, 91], vals, marker="o", label=lab(seg))
    plt.xlabel("Horizonte (dias)"); plt.ylabel("Capability efficiency proxy")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(FIG / "sensitivity_efficiency.png", dpi=110); plt.close()
    print("   fig: sensitivity_efficiency.png")
except Exception as e:
    print("  aviso fig eff:", e)
# 2) bootstrap %
try:
    plt.figure(figsize=(8, 5))
    for i, seg in enumerate(CANDIDATES):
        r = boot_df[boot_df["candidato"] == lab(seg)].iloc[0]
        lo = (r["cluster_median"] - r["cluster_P2_5"]) * 100
        hi = (r["cluster_P97_5"] - r["cluster_median"]) * 100
        plt.errorbar(r["cluster_median"] * 100, i, xerr=[[lo], [hi]], fmt="o",
                     capsize=4, label=lab(seg))
    plt.yticks(np.arange(len(CANDIDATES)), [lab(c) for c in CANDIDATES], fontsize=7)
    plt.xlabel("Capital efficiency (%) — bootstrap cluster owner")
    plt.tight_layout(); plt.savefig(FIG / "bootstrap_efficiency.png", dpi=110); plt.close()
    print("   fig: bootstrap_efficiency.png")
except Exception as e:
    print("  aviso fig boot:", e)
# 3) final tradeoffs
try:
    plt.figure(figsize=(8, 6))
    for seg in CANDIDATES:
        cap = float(inv_idx.loc[(seg[2], seg[1], seg[0])]["capital_efficiency_proxy_91"]) * 100
        pk = float(inv_idx.loc[(seg[2], seg[1], seg[0])]["net_pickup_value_median"])
        col = "#2ca02c" if stat[lab(seg)] == "ND" else "#bfbfbf"
        plt.scatter(cap, pk, c=col, s=120)
        plt.annotate(lab(seg), (cap, pk), textcoords="offset points", xytext=(6, 6), fontsize=8)
    plt.xlabel("capital_efficiency H91 (%)"); plt.ylabel("net_pickup_value")
    plt.title("Final tradeoffs (sem score)")
    plt.tight_layout(); plt.savefig(FIG / "final_candidate_tradeoffs.png", dpi=110); plt.close()
    print("   fig: final_candidate_tradeoffs.png")
except Exception as e:
    print("  aviso fig final:", e)

print("\nDONE etapa 05 (patch final)")


