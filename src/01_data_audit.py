"""
Auditoria v1.1 dos 5 datasets. Reproduzível, somente leitura de data/.
Não modifica data/. Não calcula receita. Não anualiza preço. Não usa taxa
externa de ocupação.
Gera artefatos em outputs/tables/.
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 120)

ID_COLS = {
    "Details_Itapema.csv": ["airbnb_listing_id", "owner_id"],
    "Hosts_ids_Itapema.csv": ["owner_id"],
    "Mesh_Ids_Data_Itapema.csv": ["airbnb_listing_id"],
    "Price_AV_Itapema.csv": ["airbnb_listing_id"],
    "VivaReal_Itapema.csv": ["listing_id"],
}


def load(fname: str) -> pd.DataFrame:
    dtype = {c: str for c in ID_COLS.get(fname, [])}
    df = pd.read_csv(DATA / fname, encoding="utf-8", low_memory=False,
                     dtype=dtype)
    df.columns = [c.strip() for c in df.columns]
    for c in ID_COLS.get(fname, []):
        if c in df.columns:
            df[c] = df[c].str.strip()
    return df


def parse_data(df: pd.DataFrame, cols: list) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False)
    print(f"   saved -> {OUT / name}")


print("=" * 72); print("CARREGANDO (ids lidos como string)")
details = load("Details_Itapema.csv")
hosts = load("Hosts_ids_Itapema.csv")
mesh = load("Mesh_Ids_Data_Itapema.csv")
price = load("Price_AV_Itapema.csv")
vr = load("VivaReal_Itapema.csv")
parse_data(details, ["aquisition_date"])
parse_data(hosts, ["host_snapshot_date"])
parse_data(mesh, ["aquisition_date"])
parse_data(price, ["aquisition_date", "date"])
parse_data(vr, ["aquisition_date"])

price["capture_day"] = price["aquisition_date"].dt.floor("D")
price["stay"] = price["date"].dt.normalize()
price["lead"] = (price["stay"] - price["capture_day"]).dt.days

# ============================================================
# 1. PRICE_AV
# ============================================================
print("=" * 72)
print("1. PRICE_AV")
print("   Detalhe -> Price_AV e relacao 1:N (1 listing tem varias linhas de preco).")

ts_tbl = price.groupby(["airbnb_listing_id", "capture_day", "aquisition_date"]).agg(
    n_stays=("stay", "nunique"),
    stay_min=("stay", "min"),
    stay_max=("stay", "max"),
).reset_index()
save(ts_tbl[["airbnb_listing_id", "capture_day", "aquisition_date",
             "stay_min", "stay_max", "n_stays"]], "audit_price_ts_table.csv")

# conjuntos de stay por (listing, capture_day, timestamp)
stay_sets = price.groupby(["airbnb_listing_id", "capture_day", "aquisition_date"])["stay"].apply(set)
pair_ts = ts_tbl.groupby(["airbnb_listing_id", "capture_day"])["aquisition_date"].nunique()
mts = pair_ts[pair_ts >= 2]
print(f"   pares (listing, capture_day) com >=2 timestamps: {len(mts)}")

n_pairs = 0
n_disjoint = 0
n_any_overlap = 0
for (lid, cd) in mts.index:
    chunks = stay_sets[[k for k in stay_sets.index if k[0] == lid and k[1] == cd]]
    chunk_list = [c for c in chunks.values]
    if len(chunk_list) < 2:
        continue
    n_pairs += 1
    any_overlap = False
    for a in range(len(chunk_list)):
        for b in range(a + 1, len(chunk_list)):
            if chunk_list[a] & chunk_list[b]:
                any_overlap = True
                break
        if any_overlap:
            break
    if any_overlap:
        n_any_overlap += 1
    else:
        n_disjoint += 1
print(f"   pares analisados: {n_pairs}")
print(f"   com ALGUMA sobreposicao de stay_date entre chunks: {n_any_overlap}")
print(f"   sem NENHUMA sobreposicao (conjuntos disjuntos): {n_disjoint}")
print("   => sem sobreposicao e EVIDENCIA compativel com coleta fragmentada,")
print("      NAO prova semantica definitiva de snapshots independentes.")
pd.DataFrame([{"pares_>=2ts": len(mts), "pares_analisados": n_pairs,
               "com_alguma_sobreposicao": n_any_overlap,
               "sem_sobreposicao": n_disjoint}]).to_csv(
    OUT / "audit_price_ts_overlap.csv", index=False)
print("   saved -> audit_price_ts_overlap.csv")

# ============================================================
# 2. HOSTS
# ============================================================
print("=" * 72); print("2. HOSTS")
dup_key = hosts.duplicated(subset=["owner_id", "host_snapshot_date"], keep=False)
print("   unicidade (owner_id, host_snapshot_date): linhas em chave repetida ->", int(dup_key.sum()))
dup_hosts = hosts[dup_key]
print("   combinacoes (owner_id,snapshot) duplicadas:", dup_hosts.groupby(["owner_id", "host_snapshot_date"]).ngroups)
diff_attr = 0
for (oid, snap), grp in dup_hosts.groupby(["owner_id", "host_snapshot_date"]):
    others = grp.drop(columns=["owner_id", "host_snapshot_date"])
    if others.nunique().gt(1).any():
        diff_attr += 1
print("   grupos dup c/ >1 conjunto de atributos:", diff_attr)

hosts_sorted = hosts.sort_values("host_snapshot_date")
hosts_latest = hosts_sorted.drop_duplicates("owner_id", keep="last")
print("   regra snapshot mais recente -> linhas:", len(hosts_latest),
      "| owner_id unicos:", hosts_latest["owner_id"].nunique(),
      "| owner_id duplicado:", int(hosts_latest["owner_id"].duplicated().sum()))
save(hosts_latest, "audit_hosts_latest_per_owner.csv")

# ============================================================
# 3. RATINGS
# ============================================================
print("=" * 72); print("3. RATINGS")
ct = pd.crosstab(details["number_of_reviews"] == 0, details["star_rating"] == 0)
ct.index = ["nrev==0? False", "nrev==0? True"]
ct.columns = ["star==0? False", "star==0? True"]
print("   Details (linhas=nrev==0, colunas=star==0):")
print(ct)
save(ct.reset_index(), "audit_ratings_zero_vs_reviews.csv")

hr = hosts["number_of_reviews_host"].fillna(0) == 0
sr = hosts["star_rating_host"] == 0
cth = pd.crosstab(hr, sr)
cth.index = ["nrev_host==0? False", "nrev_host==0? True"]
cth.columns = ["star_host==0? False", "star_host==0? True"]
print("   Hosts (linhas=nrev_host==0, colunas=star_host==0):")
print(cth)
save(cth.reset_index(), "audit_ratings_host_ct.csv")

# ============================================================
# 4. TESE DOS COMPACTOS (apenas amostra/contagem)
# ============================================================
print("=" * 72); print("4. TESE DOS COMPACTOS (amostra)")
det = details[["airbnb_listing_id", "number_of_bedrooms", "listing_type"]].copy()
det["bedrooms"] = pd.to_numeric(det["number_of_bedrooms"], errors="coerce").fillna(-1)
det["tipo"] = details["listing_type"].astype(str).str.strip().str.lower()
dc = det.merge(mesh[["airbnb_listing_id", "suburb"]], on="airbnb_listing_id", how="left")
dc["bc"] = dc["suburb"].astype(str).str.strip().str.lower()
price_ids = set(price["airbnb_listing_id"])
has_price = dc["airbnb_listing_id"].isin(price_ids)

studio = dc["bedrooms"] == 0
one_bed = dc["bedrooms"] == 1
centro = dc["bc"] == "centro"
apto = dc["tipo"] == "apartamento"

def compact_row(mask, label):
    sub = dc[mask]
    n = len(sub)
    nc = int(sub["airbnb_listing_id"].isin(price_ids).sum())
    return {"perfil": label, "n_total": n,
            "n_com_price": nc,
            "cobertura_percentual": round(100 * nc / n, 1) if n else 0.0}

compact = pd.DataFrame([
    compact_row(studio & centro, "studio (bedrooms==0) + Centro"),
    compact_row(one_bed & centro, "1 quarto + Centro"),
    compact_row(one_bed & centro & apto, "apartamento + 1 quarto + Centro"),
])
print(compact)
save(compact, "audit_compact_thesis_sample.csv")

# incoerida: quantos studios em toda a base? (contexto)
print("   (contexto) studios total em Details:", int(studio.sum()),
      "| studios Centro:", int((studio & centro).sum()))

# ============================================================
# 5. COBERTURA (denominador = TODOS os listings de Details)
# ============================================================
print("=" * 72); print("5. COBERTURA (Details como denominador)")
pid = set(price["airbnb_listing_id"].unique())
px = details[["airbnb_listing_id", "number_of_bedrooms", "listing_type"]].copy()
px["has_price"] = px["airbnb_listing_id"].isin(pid)
px = px.merge(mesh[["airbnb_listing_id", "suburb"]], on="airbnb_listing_id", how="left")

def cov_frame(df, col):
    g = df.groupby(col, dropna=False)["has_price"].agg(["size", "sum"])
    g = g.rename(columns={"size": "n_listings", "sum": "n_com_price"})
    g["cobertura_%"] = (100 * g["n_com_price"] / g["n_listings"]).round(1)
    g = g.reset_index().rename(columns={col: "categoria"})
    g["categoria"] = ["Missing/Unknown" if (v is None or (isinstance(v, float) and pd.isna(v)))
                      else str(v) for v in g["categoria"]]
    return g

for col in ["suburb", "number_of_bedrooms", "listing_type"]:
    g = cov_frame(px, col)
    print(f"   cobertura por {col}:")
    print(g.sort_values("n_listings", ascending=False))
    save(g, f"audit_coverage_by_{col}.csv")

# ============================================================
# 6. VIVAREAL
# ============================================================
print("=" * 72); print("6. VIVAREAL")

def dup_report(df, subset=None):
    keep_false = int(df.duplicated(subset=subset, keep=False).sum())
    # linhas redundantes removiveis = ocorrencias posteriores (drop da 2a em diante)
    first = df.duplicated(subset=subset, keep="first")
    # contar quantas linhas estao "redundantes"= ocorre apos a 1a de cada grupo dup (soma da mascara first True)
    redundant = int(df.duplicated(subset=subset, keep="first").sum())
    return keep_false, redundant

fx_all, fx_red = dup_report(vr, subset=None)
fx_key, fx_key_red = dup_report(vr, subset=["listing_id"])
print(f"   duplicatas exatas (todas cols): '{fx_all}' linhas em maior acometidas(keep=False);"
      f" '{fx_red}' redundantes/removiveis (keep='first')")
print(f"   listing_id repetido: '{fx_all}'? ' vs linhas redundantes por listing_id: '{fx_key_red}'")
print(f"   (duplicatas por listing_id - keep=False): {fx_key}")
print(f"   (duplicatas por listing_id - keep='first' = removiveis): {fx_key_red}")

num_price = pd.to_numeric(vr["sale_price"], errors="coerce")
num_area = pd.to_numeric(vr["usable_area"], errors="coerce")

print("   --- 20 maiores usable_area (suspeitas p/ revisao; nao removidas):")
print(vr.loc[num_area.nlargest(20).index,
            ["listing_id", "suburb", "usable_area", "sale_price", "bedrooms", "link_url"]].to_string())
print("   --- 20 maiores sale_price:")
print(vr.loc[num_price.nlargest(20).index,
            ["listing_id", "suburb", "usable_area", "sale_price", "bedrooms", "link_url"]].to_string())
print("   --- 20 menores sale_price:")
print(vr.loc[num_price.nsmallest(20).index,
            ["listing_id", "suburb", "usable_area", "sale_price", "bedrooms", "link_url"]].to_string())

# tabela de suburb: valores originais + freq + canonical_candidate vazia
suburb_tbl = (vr["suburb"].astype(str).str.strip()
              .replace({"nan": "", "<NA>": ""})
              .value_counts()
              .rename_axis("suburb_original")
              .reset_index(name="frequencia"))
suburb_tbl["canonical_candidate"] = ""
if "" in suburb_tbl["suburb_original"].values:
    suburb_tbl.loc[suburb_tbl["suburb_original"] == "", "suburb_original"] = "Missing/Unknown"
print("   suburbios distintos:", suburb_tbl["suburb_original"].nunique(),
      "| (nota: usado str.strip; nao e normalizacao definitiva)")
save(suburb_tbl, "audit_vivareal_suburb_mapping.csv")
print("   (tabela de mapeamento suburb -> outputs/tables/audit_vivareal_suburb_mapping.csv)")

# ============================================================
# 7. ASSERTIONS (propriedades criticas)
# ============================================================
print("=" * 72); print("7. ASSERTIONS")
checks = []
def check(name, cond):
    checks.append((name, bool(cond)))
    status = "OK" if cond else "FALHOU"
    print(f"   [{'OK' if cond else 'FALHA'}] {name}")

check("Details.airbnb_listing_id unico",
      details["airbnb_listing_id"].is_unique)
check("Mesh.airbnb_listing_id unico",
      mesh["airbnb_listing_id"].is_unique)
check("sem duplicacao (listing, capture_day, stay) no Price_AV",
      not price.duplicated(subset=["airbnb_listing_id", "capture_day", "stay"]).any())
check("capture_day apenas 06,07,20/01/2025",
      set(price["capture_day"].dropna().dt.strftime("%Y-%m-%d").unique())
      == {"2025-01-06", "2025-01-07", "2025-01-20"})
check("lead sempre em [0,90]",
      price["lead"].dropna().between(0, 90).all())
check("hosts consolidados unicos por owner_id",
      not hosts_latest["owner_id"].duplicated().any())
check("star_rating==0 <=> number_of_reviews==0 (Details)",
      (details["star_rating"] == 0).eq(details["number_of_reviews"] == 0).all())
check("star_rating_host==0 <=> number_of_reviews_host==0 (Hosts)",
      (hosts["star_rating_host"] == 0).eq(hosts["number_of_reviews_host"].fillna(0) == 0).all())

print("   totais checks:", len(checks), "| aprovados:", sum(c for _, c in checks))

# ============================================================
# 8. SNAPSHOTS
# ============================================================
print("=" * 72); print("8. COMPARACAO ENTRE SNAPSHOTS")
snap = []
capdays = sorted(price["capture_day"].dropna().unique())
by_day = {k: price[price["capture_day"] == k] for k in capdays}
for k in capdays:
    s = by_day[k]
    snap.append({
        "capture_day": k.strftime("%Y-%m-%d"),
        "n_linhas": len(s),
        "n_listings": s["airbnb_listing_id"].nunique(),
        "n_stay_dates": s["stay"].nunique(),
        "lead_min": int(s["lead"].min()),
        "lead_max": int(s["lead"].max()),
    })
snap_df = pd.DataFrame(snap)
print(snap_df)
save(snap_df, "audit_snapshot_comparison.csv")

# intersecao de listings entre snapshots
sets_by_day = {k: set(by_day[k]["airbnb_listing_id"].unique()) for k in capdays}
from itertools import combinations
inter = []
for a, b in combinations(capdays, 2):
    inter.append({
        "snap_a": a.strftime("%Y-%m-%d"),
        "snap_b": b.strftime("%Y-%m-%d"),
        "interseccao_listings": len(sets_by_day[a] & sets_by_day[b]),
        "somente_a": len(sets_by_day[a] - sets_by_day[b]),
        "somente_b": len(sets_by_day[b] - sets_by_day[a]),
    })
inter_df = pd.DataFrame(inter)
print(inter_df)
save(inter_df, "audit_snapshot_intersection.csv")

print("\nOK - auditoria v1.1 concluida sem erro.")