"""Compare the tool's promos with promos a human picked by hand.

  python evaluate.py outputs/run-XXXX gold.json

gold.json is a list of hand-picked promos (times in seconds or "MM:SS.ss"):
  [{"episode": 1, "start": "00:11.30", "end": "00:45.00", "note": "aisle humiliation"}]

A human pick counts as FOUND if any tool promo covers at least half of it.
Reports recall (share of human picks the tool found) and, for the episodes you labelled,
precision of the tool's picks (share of tool promos that overlap a reference pick).
"""
import json
import sys
from pathlib import Path

from promo_finder.parse import to_seconds


def secs(v):
    return float(v) if isinstance(v, (int, float)) else to_seconds(str(v))


def main(run_dir, gold_path):
    promos = json.loads((Path(run_dir) / "promos_ranked.json").read_text(encoding="utf-8"))
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    labelled = {g["episode"] for g in gold}

    def cover(g, p):
        gs, ge = secs(g["start"]), secs(g["end"])
        best = 0.0
        for s in p["segments"]:
            if s["episode"] == g["episode"]:
                inter = min(ge, s["end_sec"]) - max(gs, s["start_sec"])
                best = max(best, inter / (ge - gs) if inter > 0 else 0.0)
        return best

    found = 0
    print("Reference picks:")
    for g in gold:
        hits = sorted(((cover(g, p), p) for p in promos), key=lambda x: -x[0])
        ok = hits and hits[0][0] >= 0.5
        found += bool(ok)
        best = f"{hits[0][1]['id']} (rank {hits[0][1]['rank']}, covers {hits[0][0]:.0%})" if hits and hits[0][0] > 0 else "none"
        print(f"  {'FOUND ' if ok else 'MISSED'} ep {g['episode']} {g['start']}-{g['end']} {g.get('note', '')}  -> {best}")

    in_scope = [p for p in promos if set(p["episodes"]) & labelled]
    good = [p for p in in_scope if any(cover(g, p) > 0 for g in gold)]
    print(f"\nRecall: {found}/{len(gold)} reference picks found")
    if in_scope:
        print(f"Precision: {len(good)}/{len(in_scope)} tool promos in labelled episodes overlap a reference pick")
    top10 = promos[:10]
    print(f"Top-10 tool promos overlapping a human pick: {sum(1 for p in top10 if p in good)}/10 "
          f"(only meaningful if top-10 fall in labelled episodes)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
