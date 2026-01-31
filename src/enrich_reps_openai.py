

from __future__ import annotations


import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from requests.exceptions import RequestException


import pandas as pd
import requests
import numpy as np
from tqdm import tqdm

from .io_atomic import write_parquet_atomic
from .reuse import should_skip
from .run_context import enforce_match, get_week_end

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
        """
        Robust POST with retry/backoff for transient network failures and 5xx/429.
        """
        url = f"{OPENAI_BASE}{path}"
        last_err: Optional[Exception] = None

        # Tune these without changing call sites
        max_attempts = int(os.environ.get("OPENAI_HTTP_MAX_ATTEMPTS", "6"))
        base_sleep = float(os.environ.get("OPENAI_HTTP_BASE_SLEEP_S", "1.0"))
        max_sleep = float(os.environ.get("OPENAI_HTTP_MAX_SLEEP_S", "20.0"))

        for attempt in range(1, max_attempts + 1):
            try:
                r = requests.post(
                    url,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=timeout,
                )

                # Retry on rate limits / transient server errors
                if r.status_code in (429, 500, 502, 503, 504):
                    # Respect Retry-After if provided, otherwise exponential backoff
                    retry_after = r.headers.get("Retry-After")
                    if retry_after:
                        sleep_s = min(float(retry_after), max_sleep)
                    else:
                        sleep_s = min(base_sleep * (2 ** (attempt - 1)), max_sleep)
                    time.sleep(sleep_s)
                    continue

                if r.status_code >= 400:
                    # Do not retry schema/model issues etc.
                    raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text}")

                return r.json()

            except (RequestException, OSError) as e:
                # Connection reset, TLS hiccups, etc.
                last_err = e
                if attempt == max_attempts:
                    break
                sleep_s = min(base_sleep * (2 ** (attempt - 1)), max_sleep)
                time.sleep(sleep_s)

        raise RuntimeError(f"OpenAI request failed after {max_attempts} attempts: {last_err}") from last_err

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


def embed_batch(
    client: OpenAIHTTP,
    texts: List[str],
    model: str,
    retries: int = 5,
    sleep_s: float = 1.0,
) -> List[List[float]]:
    payload = {"model": model, "input": texts}

    last_err: Optional[Exception] = None
    for k in range(retries + 1):
        try:
            out = client.post("/embeddings", payload, timeout=120)
            data = out.get("data", [])

            if len(data) != len(texts):
                raise RuntimeError(
                    f"Embedding batch size mismatch: got {len(data)} embeddings for {len(texts)} inputs"
                )

            return [d["embedding"] for d in data]

        except Exception as e:
            last_err = e

            # If the API returned a hard 4xx (other than 429), don't spin forever.
            msg = str(e)
            hard_4xx = (
                ("OpenAI HTTP 400" in msg)
                or ("OpenAI HTTP 401" in msg)
                or ("OpenAI HTTP 403" in msg)
                or ("OpenAI HTTP 404" in msg)
            )
            if hard_4xx:
                raise

            # Backoff (1,2,4,8,16...) with a cap
            delay = min(60.0, sleep_s * (2 ** k))
            print(f"[WARN] embeddings request failed (attempt {k+1}/{retries+1}): {e} ; sleeping {delay:.1f}s")
            time.sleep(delay)

    raise RuntimeError(f"OpenAI embeddings failed after retries: {last_err}") from last_err

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
    force: bool = False,
) -> Path:

    # Skip before doing anything expensive (including creating the OpenAI client)
    out_parquet = Path(out_parquet)
    meta_path = out_parquet.parent / "meta.json"
    if should_skip(out_parquet, force):
        print(f"SKIP: {out_parquet} exists and --force not set.")
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "week_end": week_end,
            "emb_model": emb_model,
            "cls_model": cls_model,
            "jaccard_threshold": jaccard_threshold,
            "max_clusters_per_symbol": max_clusters_per_symbol,
            "output": str(out_parquet),
            "cache_key": hashlib.sha256(
                f"{emb_model}|{cls_model}|{jaccard_threshold}|{max_clusters_per_symbol}".encode()
            ).hexdigest()[:16],
            "skipped": True,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Wrote meta: {meta_path}")
        return out_parquet

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

    # Handle empty input
    if len(df) == 0:
        print(f"[INFO] No clusters to process. Creating empty output.")
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        empty_df = pd.DataFrame(columns=[
            "week_ending_date", "symbol", "cluster_id", "embedding", "sentiment_json", "event_json",
            "embedding_model", "classifier_model"
        ])
        write_parquet_atomic(empty_df, out_parquet)
        # Write meta.json for empty case
        meta_path = out_parquet.parent / "meta.json"
        meta = {
            "week_end": week_end,
            "emb_model": emb_model,
            "cls_model": cls_model,
            "jaccard_threshold": jaccard_threshold,
            "max_clusters_per_symbol": max_clusters_per_symbol,
            "output": str(out_parquet),
            "cache_key": hashlib.sha256(f"{emb_model}|{cls_model}|{jaccard_threshold}|{max_clusters_per_symbol}".encode()).hexdigest()[:16],
            "skipped": False,
            "rows": 0,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Wrote: {out_parquet} (0 rows)")
        print(f"Wrote meta: {meta_path}")
        return out_parquet

    # Skip already enriched clusters if output exists
    existing = None
    if out_parquet.exists():
        existing = pd.read_parquet(out_parquet)
        if "cluster_id" in existing.columns and len(existing) > 0:
            done = set(existing["cluster_id"].astype(str).tolist())
            before = len(df)
            df = df[~df["cluster_id"].astype(str).isin(done)].copy()
            print(f"[INFO] resume: {before:,} clusters loaded, {len(done):,} already enriched, {len(df):,} remaining")
            
            # If nothing to process, we're done
            if len(df) == 0:
                print(f"[INFO] All clusters already enriched. Nothing to do.")
                return out_parquet

    for col in ["symbol","cluster_id","rep_published_utc","rep_headline","rep_summary"]:
        if col not in df.columns:
            raise RuntimeError(f"clusters.parquet missing required column: {col}")

    df["rep_text"] = df.apply(build_rep_text, axis=1)
    reps: List[str] = df["rep_text"].astype(str).tolist()

    # Build the ordered list of representative texts (1:1 aligned with df rows)
    reps: List[str] = df["rep_text"].astype(str).tolist()

    # ----------------------------
    # Embeddings: checkpoint + resume (index-based)
    # ----------------------------
    emb_ckpt = out_parquet.parent / "embeddings_checkpoint.parquet"
    embeddings: List[Optional[List[float]]] = [None] * len(reps)

    # Load checkpoint if present (and not forcing rebuild)
    if emb_ckpt.exists() and not force:
        try:
            ck = pd.read_parquet(emb_ckpt)
            # expected columns: idx, embedding
            if "idx" in ck.columns and "embedding" in ck.columns and len(ck) > 0:
                for _, r in ck.iterrows():
                    j = int(r["idx"])
                    if 0 <= j < len(embeddings):
                        embeddings[j] = r["embedding"]
            print(f"[INFO] loaded embeddings checkpoint: {emb_ckpt} ({ck.shape[0]} rows)")
        except Exception as e:
            print(f"[WARN] failed to load embeddings checkpoint {emb_ckpt}: {e}")

    def _write_emb_ckpt(vecs: List[Optional[List[float]]]) -> None:
        rows = [(i, v) for i, v in enumerate(vecs) if v is not None]
        if not rows:
            return
        ckdf = pd.DataFrame(rows, columns=["idx", "embedding"])
        emb_ckpt.parent.mkdir(parents=True, exist_ok=True)
        write_parquet_atomic(ckdf, emb_ckpt)

    # Resume at first missing index
    start_i = 0
    while start_i < len(embeddings) and embeddings[start_i] is not None:
        start_i += 1

    if start_i == len(embeddings):
        print("[INFO] embeddings already complete from checkpoint")
    else:
        for i in tqdm(range(start_i, len(reps), emb_batch_size), desc="embeddings"):
            # Find next contiguous missing run starting at i
            if embeddings[i] is not None:
                continue
            batch = reps[i : i + emb_batch_size]
            batch_emb = embed_batch(client, batch, emb_model)

            # Assign back into the embeddings list
            for k, v in enumerate(batch_emb):
                embeddings[i + k] = v

            # Checkpoint every batch (small + safe; can tune later)
            _write_emb_ckpt(embeddings)

    # Final sanity check
    if any(v is None for v in embeddings):
        missing = sum(1 for v in embeddings if v is None)
        raise RuntimeError(f"Embeddings incomplete: missing {missing} of {len(embeddings)} vectors")

    # Type-narrow to the expected downstream type
    embeddings = [v for v in embeddings if v is not None]

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
    write_parquet_atomic(out, out_parquet)
    meta = {
        "week_end": week_end,
        "emb_model": emb_model,
        "cls_model": cls_model,
        "jaccard_threshold": jaccard_threshold,
        "max_clusters_per_symbol": max_clusters_per_symbol,
        "output": str(out_parquet),
        "cache_key": hashlib.sha256(
            f"{emb_model}|{cls_model}|{jaccard_threshold}|{max_clusters_per_symbol}".encode()
        ).hexdigest()[:16],
        "skipped": False,
        "rows": int(len(out)),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote: {out_parquet} ({len(out):,} rows)")
    print(f"Wrote meta: {meta_path}")
    return out_parquet




if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        "Enrich representative clusters with embeddings + OpenAI sentiment/event classification"
    )
    ap.add_argument(
        "--week_end",
        required=False,
        default=None,
        help="Week ending date YYYY-MM-DD (optional; normally from env)",
    )
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--emb_model", default="text-embedding-3-small")
    ap.add_argument("--cls_model", default="gpt-4o-mini")
    ap.add_argument("--emb_batch_size", type=int, default=128)
    ap.add_argument("--sleep_s", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="Limit number of clusters for testing")
    ap.add_argument("--max_clusters_per_symbol", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = ap.parse_args()

    canonical = get_week_end(args.week_end)
    enforce_match(args.week_end, canonical)
    week_end_str = canonical.isoformat()

    clusters = args.clusters or f"data/derived/news_clusters/week_ending={week_end_str}/clusters.parquet"
    outp = args.out or f"data/derived/rep_enriched/week_ending={week_end_str}/rep_enriched.parquet"

    run(
        week_end=week_end_str,
        clusters_parquet=Path(clusters),
        out_parquet=Path(outp),
        emb_model=args.emb_model,
        cls_model=args.cls_model,
        emb_batch_size=args.emb_batch_size,
        sleep_s=args.sleep_s,
        limit=args.limit,
        max_clusters_per_symbol=args.max_clusters_per_symbol,
        force=args.force,
    )
