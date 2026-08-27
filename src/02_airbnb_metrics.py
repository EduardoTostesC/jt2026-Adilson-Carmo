"""
Métricas de performance Airbnb — PROXY. Snapshot principal 20/01/2025.

Decisão metodológica (Price_AV = calendário de PREÇOS DISPONÍVEIS):
  - linha existente                   => data disponível com preço anunciado;
  - ausência de stay em D..D+90       => INDISPONÍVEL (PROXY);
  - indisponível NÃO = reserva (bloqueio/manutenção/uso do proprietário);
  - listing ausente do Price_AV       => estado desconhecido (não 100% indisponível).

Revisão final:
  static_revenue_proxy_H = median_available_price_H × unavailable_nights_H
    (pressupõe que a mediana das noites ainda disponíveis aproxima o valor das
     indisponíveis). NUNCA é receita observada.
  pickup = segunda evidência independente e temporal; nunca chame de reservas.

FAZ: proxies por listing/horizonte, pickup normalizado, painel, segmentos,
     tese, robustez entre snapshots.
NÃO faz: VivaReal, ROI, anualização, ML, escolha de imóvel, alteração de data/.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

PD = pd.Timestamp
MAIN_DAY = PD("2025-01-20")
D06 = PD("2025-01-06")
D07 = PD("2025-01-07")
DAYS = [D06, D07, MAIN_DAY]
HORIZONS = [30, 60, 91]
COMMON_DATES = pd.date_range(MAIN_DAY, D06 + pd.Timedelta(days=90), freq="D")


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False)
    print(f"  saved -> outputs/tables/{name} (rows={len(df)})")


def norma(val):
    return "Missing/Unknown" if (val is None or (isinstance(val, float) and pd.isna(val))) else val


def faixa(n):
    if n < 5:
        return "critico_<5"
    if n < 20:
        return "pequeno_5-19"
    if n < 50:
        return "medio_20-49"
    return "grande_50+"


# ---------------- carga ----------------
details = pd.read_csv(DATA / "Details_Itapema.csv", dtype={"airbnb_listing_id": str, "owner_id": str})
mesh = pd.read_csv(DATA / "Mesh_Ids_Data_Itapema.csv", dtype={"airbnb_listing_id": str})
price = pd.read_csv(DATA / "Price_AV_Itapema.csv", dtype={"airbnb_listing_id": str})
price["aquisition_date"] = pd.to_datetime(price["aquisition_date"], errors="coerce")
price["capture_day"] = price["aquisition_date"].dt.floor("D")
price["stay"] = pd.to_datetime(price["date"], errors="coerce").dt.normalize()
price["lead"] = (price["stay"] - price["capture_day"]).dt.days
price["price_num"] = pd.to_numeric(price["price"], errors="coerce")

print("=" * 78)
print("REPRO — PROPRIEDADES Price_AV")
assert price["capture_day"].dropna().nunique() == 3, "captures != 3"
for d in DAYS:
    s = price[price["capture_day"] == d]
    assert s["lead"].between(0, 90).all(), f"lead fora de [0,90] em {d:%Y-%m-%d}"
    assert not s.duplicated(subset=["airbnb_listing_id", "stay"]).any(), f"dup em {d:%Y-%m-%d}"
caps = {d: set(price.loc[price["capture_day"] == d, "airbnb_listing_id"].unique()) for d in DAYS}
print("  listings por capture:", {d.strftime("%Y-%m-%d"): len(caps[d]) for d in DAYS})
print("  interseção 06×20:", len(caps[D06] & caps[MAIN_DAY]),
      "| interseção 3:", len(caps[D06] & caps[D07] & caps[MAIN_DAY]))
snapshot = caps[MAIN_DAY]

# ============================================================
# 1. métricas por listing/horizonte (20/01)
# ============================================================
print("=" * 78); print("OBJETIVO 1 — métricas por listing (H=30,60,91)")
main = price[price["capture_day"] == MAIN_DAY].copy()
rows = []
for lid, sub in main.groupby("airbnb_listing_id"):
    for H in HORIZONS:
        s = sub[(sub["lead"] >= 0) & (sub["lead"] < H)]
        avail = len(s)
        unavail = H - avail
        rate = unavail / H
        pr = s["price_num"].dropna()
        n_price = int(len(pr))
        med = float(pr.median()) if n_price else np.nan
        mean = float(pr.mean()) if n_price else np.nan
        p25 = float(pr.quantile(0.25)) if n_price else np.nan
        p75 = float(pr.quantile(0.75)) if n_price else np.nan
        rows.append({
            "airbnb_listing_id": lid, "horizon": H,
            "available_nights_H": avail, "unavailable_nights_H": unavail,
            "unavailability_rate_H": rate,
            "median_available_price_H": med, "mean_available_price_H": mean,
            "p25_price_H": p25, "p75_price_H": p75,
            "n_prices_observed_H": n_price,
            "static_rev_px_per_night_H": (med * rate if n_price else np.nan),
            "static_revenue_proxy_H": (med * unavail if n_price else np.nan),
        })
metrics = pd.DataFrame(rows)
metrics["is_in_price_av_20jan"] = metrics["airbnb_listing_id"].isin(snapshot).astype(int)
metrics["has_price_in_horizon"] = (metrics["n_prices_observed_H"] > 0).astype(int)
metrics["flag_no_price_in_horizon"] = (
    (metrics["is_in_price_av_20jan"] == 1) & (metrics["has_price_in_horizon"] == 0)).astype(int)

W = metrics.pivot_table(index="airbnb_listing_id", columns="horizon",
                        values=["available_nights_H", "unavailable_nights_H",
                                "unavailability_rate_H", "median_available_price_H",
                                "mean_available_price_H", "p25_price_H", "p75_price_H",
                                "n_prices_observed_H", "static_revenue_proxy_H"],
                        aggfunc="first")
W.columns = [f"{v}_{int(k)}" for v, k in W.columns]
W = W.reset_index()
for H in HORIZONS:
    W[f"has_price_H_{H}"] = W[f"n_prices_observed_H_{H}"].fillna(0) > 0

print("  --- cobertura por horizonte (20/01) ---")
snapshot_in_details = snapshot & set(details["airbnb_listing_id"])
for H in HORIZONS:
    seg = metrics[metrics["horizon"] == H]
    n_with = int((seg["n_prices_observed_H"] > 0).sum())
    n_without = int((seg["n_prices_observed_H"] == 0).sum())
    print(f"   H={H}: snapshot={len(snapshot)} Detalhes-conectáveis={len(snapshot_in_details)} "
          f"| com preço={n_with} sem preço={n_without}")
print("  NOTA: listing pode estar no snapshot mas ter 0 datas disponíveis em H "
      "=> sem preço => static proxy NaN, SEM imputação, flag levantada.")

# ============================================================
# 4. master (Details + Mesh 1:1 + métricas wide)
# ============================================================
print("=" * 78); print("OBJETIVO 4 — master (Details+Mesh, 1:1)")
numc = ["number_of_bedrooms", "number_of_bathrooms", "number_of_guests",
        "number_of_beds", "number_of_reviews", "cleaning_fee", "star_rating"]
det_cols = ["airbnb_listing_id", "listing_type"] + numc
master_df = details[[c for c in det_cols if c in details.columns]].copy()
master_df = master_df.merge(mesh[["airbnb_listing_id", "suburb"]], on="airbnb_listing_id", how="inner")
assert master_df["airbnb_listing_id"].is_unique, "master não 1:1"
for c in numc:
    master_df[c] = pd.to_numeric(master_df[c], errors="coerce")
master_df["star_rating_original"] = master_df["star_rating"]
master_df["star_rating_clean"] = np.where(
    (master_df["star_rating"].fillna(-1) == 0) & (master_df["number_of_reviews"].fillna(-1) == 0),
    np.nan, master_df["star_rating"])
master_df = master_df.merge(W, on="airbnb_listing_id", how="left")
master_df["is_in_price_av_20jan"] = master_df["airbnb_listing_id"].isin(snapshot).astype(int)
master_df["is_balanced_panel"] = master_df["airbnb_listing_id"].isin(
    caps[D06] & caps[D07] & caps[MAIN_DAY]).astype(int)
save(master_df, "listing_metrics.csv")

# ============================================================
# 3. painel balanceado
# ============================================================
print("=" * 78); print("OBJETIVO 3 — painel balanceado")
inter3 = caps[D06] & caps[D07] & caps[MAIN_DAY]
bal_sum = pd.DataFrame({"dia": [d.strftime("%Y-%m-%d") for d in DAYS],
                        "n_listings": [len(caps[d]) for d in DAYS]})
bal_sum.loc[len(bal_sum)] = ["intersecao_3_snapshots", len(inter3)]
save(bal_sum, "balanced_panel_summary.csv")
save(pd.DataFrame({"airbnb_listing_id": sorted(inter3)}), "balanced_panel.csv")

# ============================================================
# 2. booking-pickup proxy (06/01 -> 20/01)
# ============================================================
print("=" * 78); print("OBJETIVO 2 — booking-pickup proxy (06/01 -> 20/01)")
a06 = price[price["capture_day"] == D06]
a20 = price[price["capture_day"] == MAIN_DAY]
universe = caps[D06] & caps[MAIN_DAY]  # presença nos 2, sem exigir linha na janela
print(f"  universo = 06∩20 = {len(universe)} (janela {COMMON_DATES.min().date()} a {COMMON_DATES.max().date()}, {len(COMMON_DATES)} datas)")


def avail_map(df, lid):
    sub = df[(df["airbnb_listing_id"] == lid) & (df["stay"].isin(COMMON_DATES))]
    return set(sub["stay"]), sub.set_index("stay")["price_num"]


nwin = len(COMMON_DATES)
pick = []
for lid in universe:
    s06, p06 = avail_map(a06, lid)
    s20, p20 = avail_map(a20, lid)
    a2u = len(s06 - s20)
    u2a = len(s20 - s06)
    net = a2u - u2a
    pv = float(p06.loc[list(s06 - s20)].sum()) if (s06 - s20) else 0.0
    rv = float(p20.loc[list(s20 - s06)].sum()) if (s20 - s06) else 0.0
    pick.append({"airbnb_listing_id": lid,
                 "available_to_unavailable_nights": a2u,
                 "unavailable_to_available_nights": u2a,
                 "net_unavailability_pickup": net,
                 "available_to_unavailable_rate": a2u / nwin,
                 "unavailable_to_available_rate": u2a / nwin,
                 "net_transition_rate": net / nwin,
                 "pickup_value_proxy": pv, "release_value_proxy": rv,
                 "net_pickup_value_proxy": pv - rv, "n_common_dates": nwin})
pick_df = pd.DataFrame(pick)
save(pick_df, "pickup_metrics.csv")

pickw = pick_df.merge(master_df[["airbnb_listing_id", "suburb", "number_of_bedrooms", "listing_type"]],
                      on="airbnb_listing_id", how="left")
pickw["suburb"] = pickw["suburb"].apply(norma)
pickw["bedr"] = pickw["number_of_bedrooms"].apply(norma)
pickw["typ"] = pickw["listing_type"].apply(norma)


def agg_pick(df, group_cols):
    out = []
    for keys, g in df.groupby(group_cols, dropna=False):
        key = keys if isinstance(keys, tuple) else (keys,)
        rec = {"N_listings": len(g)}
        for k, c in zip(group_cols, key):
            rec[k] = c
        for mname, series in [("net_pickup_value_proxy", g["net_pickup_value_proxy"]),
                              ("net_transition_rate", g["net_transition_rate"]),
                              ("net_pickup_nights", g["net_unavailability_pickup"])]:
            series = series.dropna()
            rec[f"{mname}_sum"] = series.sum() if len(series) else np.nan
            rec[f"{mname}_mean"] = series.mean() if len(series) else np.nan
            rec[f"{mname}_median"] = series.median() if len(series) else np.nan
            rec[f"{mname}_p25"] = series.quantile(0.25) if len(series) else np.nan
            rec[f"{mname}_p75"] = series.quantile(0.75) if len(series) else np.nan
        rec["faixa_amostral"] = faixa(len(g))
        out.append(rec)
    return pd.DataFrame(out)


save(agg_pick(pickw, ["suburb"]), "pickup_by_suburb.csv")
save(agg_pick(pickw, ["suburb", "bedr", "typ"]), "pickup_by_segment.csv")

# ============================================================
# 5. segmento por horizonte (tabela longa H=30,60,91)
# ============================================================
print("=" * 78); print("OBJETIVO 5 — segmentos por horizonte (long)")
mseg = metrics.merge(master_df[["airbnb_listing_id", "suburb", "number_of_bedrooms", "listing_type"]],
                     on="airbnb_listing_id", how="inner")
mseg["suburb"] = mseg["suburb"].apply(norma)
mseg["bedr"] = mseg["number_of_bedrooms"].fillna(-1)
mseg["typ"] = mseg["listing_type"].apply(norma)
seg_rows = []
for (H, sub, bd, tp), g in mseg.groupby(["horizon", "suburb", "bedr", "typ"], dropna=False):
    n_total = len(g)
    gc = g[g["flag_no_price_in_horizon"] == 0]
    rec = {"horizon": int(H), "suburb": sub, "number_of_bedrooms": bd,
           "listing_type": tp, "N_total": n_total, "N_com_proxy": len(gc)}
    for label, col in [("ADR", "median_available_price_H"),
                       ("unavailability", "unavailability_rate_H"),
                       ("static_rev", "static_revenue_proxy_H")]:
        s = gc[col].dropna()
        rec[f"{label}_median"] = s.median() if len(s) else np.nan
        rec[f"{label}_mean"] = s.mean() if len(s) else np.nan
        rec[f"{label}_p25"] = s.quantile(0.25) if len(s) else np.nan
        rec[f"{label}_p75"] = s.quantile(0.75) if len(s) else np.nan
    rec["faixa_amostral"] = faixa(n_total)
    seg_rows.append(rec)
segments = pd.DataFrame(seg_rows)
save(segments, "segment_by_horizon.csv")

# ----- reconciliação do ranking por BAIRRO (agregando todos os quartos/tipos) -----
def suburb_rank(df, H):
    g = df[(df["horizon"] == H)].copy()
    sub = g.groupby("suburb").agg(
        N_total=("N_total", "sum"),
        N_com_proxy=("N_com_proxy", "sum")).reset_index()
    # medianas/quantis por bairro direto das métricas por listing
    mm = metrics[metrics["horizon"] == H].merge(
        master_df[["airbnb_listing_id", "suburb"]], on="airbnb_listing_id", how="inner")
    mm["suburb"] = mm["suburb"].apply(norma)
    mm = mm[mm["flag_no_price_in_horizon"] == 0]
    agg = mm.groupby("suburb")["median_available_price_H"].agg(["median", "mean"])
    agg.columns = ["ADR_median", "ADR_mean"]
    agg_unv = mm.groupby("suburb")["unavailability_rate_H"].agg("median").rename("unavailability_median")
    agg_rev = mm.groupby("suburb")["static_revenue_proxy_H"].agg(
        ["median", "mean", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    agg_rev.columns = ["rev_median", "rev_mean", "rev_p25", "rev_p75"]
    out = sub.merge(agg, left_on="suburb", right_index=True).merge(
        agg_unv, left_on="suburb", right_index=True).merge(
        agg_rev, left_on="suburb", right_index=True)
    out["horizon"] = H
    return out


suburban_all = pd.concat([suburb_rank(segments, H) for H in HORIZONS], ignore_index=True)

main_bairros = ["Meia Praia", "Centro", "Morretes", "Tabuleiro dos Oliveiras"]
for H in HORIZONS:
    four = suburban_all[(suburban_all["horizon"] == H) & (suburban_all["suburb"].isin(main_bairros))].sort_values("rev_median", ascending=False)
    print(f"  Ranking bairro H={H} (por static_rev_median):")
    print(four[["suburb", "N_total", "N_com_proxy", "ADR_median", "unavailability_median",
                "rev_median", "rev_mean", "rev_p25", "rev_p75"]].to_string(index=False))
print("  NOTA: N somado sobre combos; medianas por listing calculadas direto (não peso por combo).")

# ============================================================
# 6. tese dos compactos
# ============================================================
print("=" * 78); print("OBJETIVO 6 — tese dos compactos")
mar = master_df.copy()
mar["suburb_low"] = mar["suburb"].astype(str).str.strip().str.lower()
mar["tipo_low"] = mar["listing_type"].astype(str).str.strip().str.lower()
BR0 = pd.to_numeric(mar["number_of_bedrooms"], errors="coerce").astype(float) == 0
BR1 = pd.to_numeric(mar["number_of_bedrooms"], errors="coerce").astype(float) == 1
CENTRO = mar["suburb_low"] == "centro"
APTO = mar["tipo_low"] == "apartamento"
any_caps = caps[D06] | caps[D07] | caps[MAIN_DAY]


def thesis_row(mask, label):
    sub = mar[mask]
    n_det = len(sub)
    n_any = int(sub["airbnb_listing_id"].isin(any_caps).sum())
    n_snap = int(sub["airbnb_listing_id"].isin(snapshot).sum())
    n_proc = int((sub["n_prices_observed_H_91"].fillna(0) > 0).sum())
    gc = sub[sub["n_prices_observed_H_91"].fillna(0) > 0]
    adr = gc["median_available_price_H_91"].median() if len(gc) else np.nan
    srp = gc["static_revenue_proxy_H_91"].median() if len(gc) else np.nan
    cls = "evidência_insuficiente" if (n_det < 5 or n_any < 5 or n_proc < 5) else "amostra_suficiente"
    return {"perfil": label, "N_details": n_det, "N_any_price_av": n_any,
            "N_snapshot_20jan": n_snap, "N_com_proxy_H": n_proc,
            "ADR_median_91": adr, "static_rev_proxy_median_91": srp,
            "classificacao": cls}


thesis = pd.DataFrame([
    thesis_row(BR0 & CENTRO, "studio+Centro"),
    thesis_row(BR1 & CENTRO, "1quart+Centro"),
    thesis_row(APTO & BR1 & CENTRO, "apart+1quart+Centro"),
])
save(thesis, "compact_thesis.csv")
print(thesis.to_string(index=False))

# ============================================================
# 7. robustez entre snapshots (H=91 de cada um)
# ============================================================
print("=" * 78); print("OBJETIVO 7 — robustez entre snapshots")
rob_rows = []
for d in DAYS:
    sub_p = price[price["capture_day"] == d]
    tmp = {}
    for lid, sg in sub_p.groupby("airbnb_listing_id"):
        s = sg[(sg["lead"] >= 0) & (sg["lead"] < 91)]
        pr = s["price_num"].dropna()
        tmp[lid] = {"ADR": pr.median() if len(pr) else np.nan,
                    "unav": (91 - len(s)) / 91 if len(s) else np.nan,
                    "rev": (pr.median() * (91 - len(s))) if len(pr) else np.nan}
    rt = pd.DataFrame(tmp).T.reset_index().rename(columns={"index": "airbnb_listing_id"})
    rt = rt.merge(master_df[["airbnb_listing_id", "suburb", "number_of_bedrooms", "listing_type"]],
                  on="airbnb_listing_id", how="left")
    rt["suburb_low"] = rt["suburb"].apply(norma).astype(str).str.lower()
    rt["bedr"] = rt["number_of_bedrooms"].fillna(-1)
    for group_cols, dim in [(["suburb_low"], "suburb_low"), (["bedr"], "bedrooms"),
                            (["suburb_low", "bedr", "listing_type"], "full")]:
        gg = rt.groupby(group_cols, dropna=False).agg(N=("airbnb_listing_id", "count"),
                                                      ADR=("ADR", "median"),
                                                      unav=("unav", "median"),
                                                      rev=("rev", "median")).reset_index()
        gg["dim"] = dim
        gg["snapshot"] = d.strftime("%Y-%m-%d")
        rob_rows.append(gg)
robust = pd.concat(rob_rows, ignore_index=True)
robust["faixa_amostral"] = robust["N"].apply(faixa)
save(robust, "snapshot_robustness.csv")

print("  top6 bairro por ADR (N registrado):")
for d in DAYS:
    g = robust[(robust["dim"] == "suburb_low") & (robust["snapshot"] == d.strftime("%Y-%m-%d"))].sort_values("ADR", ascending=False).head(6)
    print(f"  {d:%m-%d}:")
    for _, r in g.iterrows():
        print(f"     {r['suburb_low']:>24} ADR={r['ADR']:.0f} unav={r['unav']:.3f} rev={r['rev']:.0f} [N={int(r['N'])}]")

print("\nDONE")