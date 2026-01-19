from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

OPENAI_BASE = "https://api.openai.com/v1"

@dataclass
class OpenAIHTTP:
    api_key: str

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def post(self, path: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        r = requests.post(f"{OPENAI_BASE}{path}", headers=self.headers, data=json.dumps(payload), timeout=timeout)
        if r.status_code >= 400:
            # Show the exact OpenAI error payload to debug schema/model issues
            raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text}")
        return r.json()

def extract_output_text(resp: Dict[str, Any]) -> str:
    parts: List[str] = []
    for item in resp.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text" and "text" in c:
                parts.append(c["text"])
    return "\n".join(parts).strip()

def build_rep_text(row: pd.Series) -> str:
    h = (row.get("rep_headline", "") or "").strip()
    s = (row.get("rep_summary", "") or "").strip()
    return (h + " || " + s).strip()

COMBINED_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sentiment": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "sent_label": {"type": "string", "enum": ["Positive", "Neutral", "Negative", "Mixed/Unclear"]},
                "sent_score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "sent_driver": {"type": "string", "enum": ["Fundamental","Capital structure","Operational","Governance","Market/Price-action","Speculation/Opinion"]},
                "rationale": {"type": "string"}
            },
            "required": ["sent_label","sent_score","confidence","sent_driver","rationale"]
        },
        "event": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "event_type_primary": {
                    "type": "string",
                    "enum": [
                        "EARNINGS_GUIDANCE","MNA_STRATEGIC","REGULATORY_LEGAL","CAPITAL_STRUCTURE",
                        "OPERATIONS_SUPPLY","PRODUCT_MARKET","MANAGEMENT_GOVERNANCE","ANALYST_ACTION",
                        "MACRO_SECTOR","PRICE_ACTION_RECAP","OTHER_LOW_SIGNAL"
                    ]
                },
                "event_severity": {"type": "integer", "minimum": 0, "maximum": 3},
                "event_direction": {"type": "string", "enum": ["Positive","Negative","Mixed/Unclear","Neutral"]},
                "event_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "rationale": {"type": "string"}
            },
            "required": ["event_type_primary","event_severity","event_direction","event_confidence","rationale"]
        }
    },
    "required": ["sentiment","event"]
}
COMBINED_SYSTEM = (
    "You are a conservative financial news classifier for a weekly research pipeline.\n"
    "Task: produce BOTH a sentiment assessment (directional impact on fundamentals, not stock price) "
    "and an event classification (type/severity/direction).\n"
    "Rules:\n"
    "- If the item is mostly a price-move recap with no new info: "
    "sent_driver='Market/Price-action', sent_score near 0, and event_type_primary='PRICE_ACTION_RECAP' with severity 0-1.\n"
    "- Be conservative: prefer Neutral/Mixed when unclear.\n"
    "- Rationales must be one sentence each (max 25 words).\n"
    "Return ONLY JSON matching the schema."
)

def resp_payload(model: str, system: str, user: str, schema_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    # Responses API + Structured Outputs (JSON Schema) goes in text.format
    # text.format must include: type, name, schema, and optional strict
    return {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_output_tokens": 300,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        }
    }


def combined_user(symbol: str, published_utc: str, headline: str, summary: str) -> str:
    return f"""Ticker: {symbol}
Published (UTC): {published_utc}
Headline: {headline}
Summary: {summary}

Return a single JSON object that matches the schema exactly.
- sentiment.rationale: one sentence (max 25 words)
- event.rationale: one sentence (max 25 words)
"""

def embed_batch(client: OpenAIHTTP, texts: List[str], model: str) -> List[List[float]]:
    payload = {"model": model, "input": texts}
    out = client.post("/embeddings", payload, timeout=120)
    data = out.get("data", [])
    return [d["embedding"] for d in data]

def classify_one(client: OpenAIHTTP, payload: Dict[str, Any], retries: int = 2, sleep_s: float = 1.0) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for k in range(retries + 1):
        try:
            resp = client.post("/responses", payload, timeout=120)
            txt = extract_output_text(resp)
            return json.loads(txt) if txt else {}
        except Exception as e:
            last_err = e
            time.sleep(sleep_s * (2 ** k))
    raise RuntimeError(f"OpenAI classify failed after retries: {last_err}") from last_err

def run(
    week_end: str,
    clusters_parquet: Path,
    out_parquet: Path,
    emb_model: str = "text-embedding-3-small",
    cls_model: str = "gpt-4o-mini",
    emb_batch_size: int = 128,
    sleep_s: float = 0.0,
    limit: int | None = None,
    jaccard_threshold: float = 0.55,
    max_clusters_per_symbol: int = 1,
) -> Path:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    client = OpenAIHTTP(key)

    if not clusters_parquet.exists():
        raise FileNotFoundError(f"Missing clusters file: {clusters_parquet}")

    df = pd.read_parquet(clusters_parquet).copy()
    print(f"[INFO] loaded clusters: {len(df):,}")
    if limit is not None:
        df = df.head(limit).copy()
    print(f"[INFO] clusters to process: {len(df):,}")

    # Skip already enriched clusters if output exists
    existing = None
    if out_parquet.exists():
        existing = pd.read_parquet(out_parquet)
        if "cluster_id" in existing.columns:
            done = set(existing["cluster_id"].astype(str).tolist())
            before = len(df)
            df = df[~df["cluster_id"].astype(str).isin(done)].copy()
            print(f"[INFO] resume: {before:,} clusters loaded, {len(done):,} already enriched, {len(df):,} remaining")

    for col in ["symbol","cluster_id","rep_published_utc","rep_headline","rep_summary"]:
        if col not in df.columns:
            raise RuntimeError(f"clusters.parquet missing required column: {col}")

    df["rep_text"] = df.apply(build_rep_text, axis=1)

    # ---- Embeddings (batched) ----
    reps = df["rep_text"].fillna("").astype(str).tolist()
    embeddings: List[List[float]] = []
    for i in tqdm(range(0, len(reps), emb_batch_size), desc="embeddings"):
        batch = reps[i:i+emb_batch_size]
        embeddings.extend(embed_batch(client, batch, emb_model))
        if sleep_s:
            time.sleep(sleep_s)

    if len(embeddings) != len(df):
        raise RuntimeError("Embedding count mismatch")

    # ---- Classification (serial v1; can parallelize later) ----
    sent_jsons = []
    event_jsons = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="classify"):
        sym = str(r["symbol"])
        pub = str(r.get("rep_published_utc", "") or "")
        head = str(r.get("rep_headline", "") or "")
        summ = str(r.get("rep_summary", "") or "")

        payload = resp_payload(
            model=cls_model,
            system=COMBINED_SYSTEM,
            user=combined_user(sym, pub, head, summ),
            schema_name="sentiment_event_schema",
            schema=COMBINED_SCHEMA,
        )

        both = classify_one(client, payload)

        # Defensive: ensure expected keys exist
        s = both.get("sentiment", {}) if isinstance(both, dict) else {}
        e = both.get("event", {}) if isinstance(both, dict) else {}

        sent_jsons.append(json.dumps(s))
        event_jsons.append(json.dumps(e))

        if sleep_s:
            time.sleep(sleep_s)


    new_out = pd.DataFrame({
        "week_ending_date": week_end,
        "symbol": df["symbol"].astype(str),
        "cluster_id": df["cluster_id"].astype(str),
        "embedding": embeddings,
        "sentiment_json": sent_jsons,
        "event_json": event_jsons,
        "embedding_model": emb_model,
        "classifier_model": cls_model,
    })

    if out_parquet.exists():
        old = pd.read_parquet(out_parquet)
        out = pd.concat([old, new_out], ignore_index=True)
        out = out.drop_duplicates(subset=["cluster_id"], keep="last")
    else:
        out = new_out

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)
    print(f"Wrote: {out_parquet} ({len(out):,} rows)")
    return out_parquet


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--week_end", required=True)
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--emb_model", default="text-embedding-3-small")
    ap.add_argument("--cls_model", default="gpt-4o-mini")
    ap.add_argument("--emb_batch_size", type=int, default=128)
    ap.add_argument("--sleep_s", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="Limit number of clusters for testing")
    ap.add_argument("--max_clusters_per_symbol", type=int, default=1)
    args = ap.parse_args()

    clusters = args.clusters or f"data/derived/news_clusters/week_ending={args.week_end}/clusters.parquet"
    outp = args.out or f"data/derived/rep_enriched/week_ending={args.week_end}/rep_enriched.parquet"

    run(
        week_end=args.week_end,
        clusters_parquet=Path(clusters),
        out_parquet=Path(outp),
        emb_model=args.emb_model,
        cls_model=args.cls_model,
        emb_batch_size=args.emb_batch_size,
        sleep_s=args.sleep_s,
        limit=args.limit,
    )
