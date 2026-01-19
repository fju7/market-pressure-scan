from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","to","of","in","on","for","with","at","by",
    "from","as","is","are","was","were","be","been","being","it","its","this","that","these","those","will","may",
    "can","could","should","would","has","have","had","new","says","said","report","reports","update","announces",
    "announced","shares","stock","market","company","inc","corp","ltd","co"
}

def normalize_text(headline: str, summary: str) -> str:
    h = (headline or "").strip()
    s = (summary or "").strip()
    # Keep headline heavily weighted
    return f"{h}. {s}".strip()

def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text or "")]
    toks = [t for t in toks if t not in STOPWORDS and len(t) >= 3]
    return toks

def token_set(text: str) -> Set[str]:
    return set(tokenize(text))

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0

def stable_cluster_id(symbol: str, rep_headline: str, rep_published_utc: str) -> str:
    base = f"{symbol}|{rep_published_utc}|{rep_headline}".encode("utf-8", errors="ignore")
    h = hashlib.sha1(base).hexdigest()[:12]
    return f"{symbol}_{h}"

@dataclass
class ClusterConfig:
    jaccard_threshold: float = 0.55
    max_docs_per_symbol: int = 5000  # safety

def run(week_end: str,
        in_parquet: Path,
        out_parquet: Path,
        jaccard_threshold: float = 0.55,
        max_clusters_per_symbol: int = 2) -> Path:
    if not in_parquet.exists():
        raise FileNotFoundError(f"Missing raw news file: {in_parquet}")

    raw = pd.read_parquet(in_parquet)

    # Expect Finnhub company-news fields; tolerate missing
    for col in ["symbol","headline","summary","url","source","published_utc","datetime"]:
        if col not in raw.columns:
            raw[col] = None

    raw["symbol"] = raw["symbol"].astype(str).str.upper().str.strip()
    if raw["published_utc"].isna().all() and "datetime" in raw.columns:
        # Finnhub provides epoch seconds in `datetime`
        raw["published_utc"] = pd.to_datetime(raw["datetime"], unit="s", utc=True, errors="coerce")
    raw["published_utc"] = pd.to_datetime(raw["published_utc"], utc=True, errors="coerce")

    # Basic dedupe: url+headline within symbol
    raw["headline"] = raw["headline"].astype(str).fillna("").str.strip()
    raw["summary"] = raw["summary"].astype(str).fillna("").str.strip()
    raw["url"] = raw["url"].astype(str).fillna("").str.strip()
    raw["source"] = raw["source"].astype(str).fillna("").str.strip()

    raw = raw.drop_duplicates(subset=["symbol","url","headline"], keep="last")
    raw = raw.sort_values(["symbol","published_utc"], ascending=[True, False])


    clusters_out = []

    for sym, g in raw.groupby("symbol"):
        g = g.head(5000).copy()

        # Precompute token sets
        g["canon"] = [normalize_text(h, s) for h, s in zip(g["headline"], g["summary"])]
        tok_sets = [token_set(t) for t in g["canon"].tolist()]

        assigned = np.zeros(len(g), dtype=bool)
        idxs = list(range(len(g)))

        sym_clusters = []

        # Greedy clustering: pick next unassigned as seed, group similar docs
        for i in idxs:
            if assigned[i]:
                continue
            seed_set = tok_sets[i]
            members = [i]
            assigned[i] = True

            for j in idxs:
                if assigned[j]:
                    continue
                if jaccard(seed_set, tok_sets[j]) >= jaccard_threshold:
                    members.append(j)
                    assigned[j] = True

            sub = g.iloc[members].copy()

            # Representative: most recent, then longest headline
            sub = sub.sort_values(["published_utc"], ascending=False)
            rep = sub.iloc[0]

            rep_published = rep["published_utc"]
            rep_published_utc = rep_published.isoformat() if pd.notna(rep_published) else ""

            rep_head = rep["headline"]
            rep_sum = rep["summary"]

            cluster_size = int(len(sub))
            unique_sources = int(sub["source"].nunique(dropna=True))

            cid = stable_cluster_id(sym, rep_head, rep_published_utc)

            sym_clusters.append({
                "week_ending_date": week_end,
                "symbol": sym,
                "cluster_id": cid,
                "rep_published_utc": rep_published_utc,
                "rep_headline": rep_head,
                "rep_summary": rep_sum,
                "cluster_size": cluster_size,
                "unique_sources": unique_sources,
            })

        if sym_clusters:
            sym_df = pd.DataFrame(sym_clusters)
            sym_df = sym_df.sort_values(["cluster_size", "unique_sources"], ascending=False).head(max_clusters_per_symbol)
            clusters_out.extend(sym_df.to_dict("records"))

    out = pd.DataFrame(clusters_out)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)
    print(f"Wrote: {out_parquet} ({len(out):,} rows)")
    return out_parquet


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--week_end", required=True)
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", dest="outp", default=None)
    ap.add_argument("--jaccard", type=float, default=0.55)
    ap.add_argument("--max_clusters_per_symbol", type=int, default=2, help="Limit clusters per symbol (post-clustering)")
    args = ap.parse_args()

    inp = args.inp or f"data/derived/news_raw/week_ending={args.week_end}/company_news.parquet"
    outp = args.outp or f"data/derived/news_clusters/week_ending={args.week_end}/clusters.parquet"

    run(args.week_end, Path(inp), Path(outp), jaccard_threshold=args.jaccard, max_clusters_per_symbol=args.max_clusters_per_symbol)
