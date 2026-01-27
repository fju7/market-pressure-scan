from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# PDF (pure python)
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def write_csv(out_csv: Path, week_end: str, meta: dict, basket_df: pd.DataFrame) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Header rows + then basket rows
    rows = []
    rows.append(["week_ending_date", week_end])
    rows.append(["generated_utc", datetime.utcnow().isoformat() + "Z"])
    rows.append(["regime", meta.get("regime", "")])
    rows.append(["schema", meta.get("schema", "")])
    rows.append(["is_low_information_week", str(meta.get("is_low_information_week", False))])
    rows.append(["recap_pct", str(meta.get("recap_pct", ""))])
    rows.append(["avg_novelty_z", str(meta.get("avg_novelty_z", ""))])
    rows.append(["avg_event_intensity_z", str(meta.get("avg_event_intensity_z", ""))])
    rows.append(["cluster_count", str(meta.get("cluster_count", ""))])

    reasons = meta.get("low_info_reasons", []) or []
    if reasons:
        rows.append(["low_info_reasons", "; ".join(reasons)])

    rows.append([])
    rows.append(["BASKET"])
    if len(basket_df) == 0:
        rows.append(["(empty)"])
    else:
        cols = basket_df.columns.tolist()
        rows.append(cols)
        for _, r in basket_df.iterrows():
            rows.append([str(r.get(c, "")) for c in cols])

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            w.writerow(row)


def write_pdf(out_pdf: Path, week_end: str, meta: dict, basket_df: pd.DataFrame) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(out_pdf), pagesize=letter)
    width, height = letter

    left = 0.75 * inch
    top = height - 0.75 * inch
    line = 14

    def draw_line(y):
        c.line(left, y, width - left, y)

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, top, f"Weekly Trader Sheet — Week Ending {week_end}")

    c.setFont("Helvetica", 9)
    c.drawString(left, top - 16, f"Generated (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    y = top - 34
    draw_line(y)
    y -= 18

    # Signal / regime summary
    is_low = bool(meta.get("is_low_information_week", False))
    recap_pct = _safe_float(meta.get("recap_pct", ""), None)
    avg_n = _safe_float(meta.get("avg_novelty_z", ""), None)
    avg_e = _safe_float(meta.get("avg_event_intensity_z", ""), None)
    cluster_count = meta.get("cluster_count", None)

    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y, "Signal Quality Snapshot")
    y -= 16

    c.setFont("Helvetica", 10)
    c.drawString(left, y, f"Regime: {meta.get('regime','')}  Schema: {meta.get('schema','')}")
    y -= line
    c.drawString(left, y, f"Low-information week: {'YES (SKIP)' if is_low else 'NO'}")
    y -= line

    if recap_pct is not None:
        c.drawString(left, y, f"PRICE_ACTION_RECAP share: {recap_pct:.0f}%")
        y -= line
    if avg_n is not None:
        c.drawString(left, y, f"Avg novelty (z): {avg_n:.2f}")
        y -= line
    if avg_e is not None:
        c.drawString(left, y, f"Avg event intensity (z): {avg_e:.2f}")
        y -= line
    if cluster_count is not None:
        c.drawString(left, y, f"Cluster count: {cluster_count}")
        y -= line

    reasons = meta.get("low_info_reasons", []) or []
    if reasons:
        y -= 6
        c.setFont("Helvetica-Bold", 10)
        c.drawString(left, y, "Reasons:")
        y -= line
        c.setFont("Helvetica", 9)
        for r in reasons[:4]:
            c.drawString(left + 12, y, f"• {r}")
            y -= 11

    y -= 10
    draw_line(y)
    y -= 18

    # Basket summary
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y, "Basket / Action")
    y -= 16

    c.setFont("Helvetica", 10)
    if len(basket_df) == 0:
        c.drawString(left, y, "No basket generated.")
        y -= line
    else:
        # If SKIP file, it will have columns: week_ending_date, action, reason
        action = str(basket_df.iloc[0].get("action", "")).upper() if "action" in basket_df.columns else ""
        if action == "SKIP":
            c.setFont("Helvetica-Bold", 12)
            c.drawString(left, y, "ACTION: SKIP (no trades this week)")
            y -= 18
            c.setFont("Helvetica", 9)
            reason = str(basket_df.iloc[0].get("reason", ""))[:200]
            if reason:
                c.drawString(left, y, f"Reason: {reason}")
                y -= line
        else:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(left, y, "ACTION: TRADE (Mon open → Fri close)")
            y -= 18

            cols = ["symbol", "sector", "UPS_adj", "conviction", "weight"]
            present = [cname for cname in cols if cname in basket_df.columns]

            # Table header
            c.setFont("Helvetica-Bold", 9)
            x_positions = [left, left + 90, left + 250, left + 335, left + 415]
            headers = ["Symbol", "Sector", "UPS_adj", "Conv", "Wt"]
            for xp, h in zip(x_positions, headers):
                c.drawString(xp, y, h)
            y -= 12
            c.setFont("Helvetica", 9)

            # Rows (max 12 to keep one-page)
            show = basket_df.copy()
            if "UPS_adj" in show.columns:
                show = show.sort_values("UPS_adj", ascending=False)
            show = show.head(12)

            for _, r in show.iterrows():
                sym = str(r.get("symbol", ""))
                sec = str(r.get("sector", ""))[:22]
                ups = r.get("UPS_adj", "")
                conv = str(r.get("conviction", ""))[:10]
                wt = r.get("weight", "")

                c.drawString(x_positions[0], y, sym)
                c.drawString(x_positions[1], y, sec)
                c.drawString(x_positions[2], y, f"{ups:.3f}" if isinstance(ups, (int, float)) else str(ups))
                c.drawString(x_positions[3], y, conv)
                c.drawString(x_positions[4], y, f"{wt:.3f}" if isinstance(wt, (int, float)) else str(wt))
                y -= 11
                if y < 2.0 * inch:
                    break

            y -= 6
            c.setFont("Helvetica", 9)
            c.drawString(left, y, "Note: Full basket is in basket.csv; PDF shows first 12 for space.")
            y -= line

    y -= 8
    draw_line(y)
    y -= 18

    # Build stamp (traceability)
    build_info = meta.get("build", {})
    if build_info:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(left, y, "Build Stamp")
        y -= 16
        c.setFont("Helvetica", 8)
        
        git_sha = build_info.get("git_sha", "")
        run_id = build_info.get("github_run_id", "")
        run_attempt = build_info.get("github_run_attempt", "")
        fs_hash = build_info.get("features_scores_sha256", "")
        
        if git_sha:
            c.drawString(left, y, f"Git SHA: {git_sha[:12]}...")
            y -= 10
        if run_id:
            c.drawString(left, y, f"GitHub Run: {run_id} (attempt {run_attempt})")
            y -= 10
        if fs_hash:
            c.drawString(left, y, f"features_scores.py: {fs_hash[:16]}...")
            y -= 10
        
        python_ver = build_info.get("python", "")
        pandas_ver = build_info.get("pandas", "")
        if python_ver:
            c.drawString(left, y, f"Python {python_ver} · pandas {pandas_ver}")
            y -= 10
        
        y -= 8
        draw_line(y)
        y -= 18

    # Ledger pointers (always)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y, "Ledger Pointers (what to update)")
    y -= 16
    c.setFont("Helvetica", 9)
    c.drawString(left, y, "After Monday open fills → append: data/live/trades_log.csv")
    y -= 12
    c.drawString(left, y, "After Friday close → append: data/live/weekly_pnl.csv")
    y -= 12
    c.drawString(left, y, f"Basket file: data/derived/baskets/week_ending={week_end}/basket.csv")
    y -= 12
    c.drawString(left, y, f"Report meta: data/derived/reports/week_ending={week_end}/report_meta.json")

    c.showPage()
    c.save()
    print(f"Wrote: {out_pdf}")


from typing import Optional

class Paths:
    def __init__(self, regime_id: str, schema_id: str, week_end: str):
        self.meta_path = Path(f"data/derived/reports/regime={regime_id}/schema={schema_id}/week_ending={week_end}/report_meta.json")
        self.basket_path = Path(f"data/derived/baskets/regime={regime_id}/schema={schema_id}/week_ending={week_end}/basket.csv")
        self.out_dir = Path(f"data/derived/trader_sheets/week_ending={week_end}")
        self.out_pdf = self.out_dir / "trader_sheet.pdf"
        self.out_csv = self.out_dir / "trader_sheet.csv"

def default_paths(regime_id: str, schema_id: str, week_end: str) -> Paths:
    return Paths(regime_id, schema_id, week_end)

def run(week_end: str, regime_id: str, schema_id: str, paths: Optional[Paths] = None):
    if paths is None:
        paths = default_paths(regime_id, schema_id, week_end)

    meta_path = paths.meta_path
    basket_path = paths.basket_path

    # Fallback to legacy paths if not found
    if not meta_path.exists():
        meta_path = Path(f"data/derived/reports/week_ending={week_end}/report_meta.json")
    if not basket_path.exists():
        basket_path = Path(f"data/derived/baskets/week_ending={week_end}/basket.csv")

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path} (run report_weekly first)")
    if not basket_path.exists():
        raise FileNotFoundError(f"Missing: {basket_path} (run export_basket first)")

    meta = _read_json(meta_path)
    basket_df = pd.read_csv(basket_path)

    out_dir = paths.out_dir
    out_pdf = paths.out_pdf
    out_csv = paths.out_csv

    write_pdf(out_pdf, week_end, meta, basket_df)
    write_csv(out_csv, week_end, meta, basket_df)
    print(f"Wrote: {out_csv}")

def main():

    from src.run_context import get_week_end, enforce_match
    ap = argparse.ArgumentParser()
    ap.add_argument("--week_end", required=False, default=None, help="Week ending date YYYY-MM-DD (optional; normally from env)")
    ap.add_argument("--regime", default="news-novelty-v1", help="Regime ID (e.g., news-novelty-v1, news-novelty-v1b)")
    ap.add_argument("--schema", default="news-novelty-v1b", help="Schema ID (e.g., news-novelty-v1, news-novelty-v1b)")
    args = ap.parse_args()

    canonical = get_week_end(args.week_end)
    enforce_match(args.week_end, canonical)

    run(week_end=canonical.isoformat(), regime_id=args.regime, schema_id=args.schema)

if __name__ == "__main__":
    main()
