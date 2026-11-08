"""Does the judge notice when a clip gets worse?  (uses the real model)

For the top-N promos of a run, build two damaged versions:
  slow-start  : starts ~12 s earlier, on setup instead of the hook
  early-cut   : ends ~40 % sooner, before the cliffhanger lands
All clips get neutral shuffled IDs, so the judge can't tell which is the original.
A useful judge should give the original a higher hook score than slow-start and a
higher cliffhanger score than early-cut.

  python tests/judge_check.py outputs/full/run-XXXX --top 12
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promo_finder import prompts as P                       # noqa: E402
from promo_finder.llm import LLM, LLMConfig                 # noqa: E402
from promo_finder.parse import load_episodes                # noqa: E402
from promo_finder.pipeline import SPOKEN, Promo, Segment, clip_script, resolve_segment  # noqa: E402
from run import load_dotenv                                 # noqa: E402


def variants(d, eps):
    s0 = d["segments"][0]
    if len(d["segments"]) != 1:
        return None
    ep = eps[s0["episode"]]
    orig = Segment(ep.number, s0["start_sec"], s0["end_sec"], s0["start_line"], s0["end_line"])
    # slow start: first line starting >= 12 s before the original start
    earlier = [l for l in ep.lines if orig.start - 14 <= l.start <= orig.start - 10]
    if not earlier:
        return None
    slow, _ = resolve_segment(ep, earlier[0].lid, orig.end_line)
    # early cut: last spoken line that ends before 60 % of the clip
    cut_at = orig.start + 0.6 * orig.duration
    before = [l for l in ep.lines if l.kind in SPOKEN and orig.start < l.end <= cut_at]
    if not before:
        return None
    early, _ = resolve_segment(ep, orig.start_line, before[-1].lid)
    return {"original": orig, "slow-start": slow, "early-cut": early}


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--model")
    a = ap.parse_args()
    run = Path(a.run_dir)
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    eps = {e.number: e for e in load_episodes(meta["input_dir"])}
    promos = [p for p in json.loads((run / "promos_ranked.json").read_text(encoding="utf-8")) if p["kind"] == "single"]

    first = sorted(eps.values(), key=lambda e: e.number)[:3]
    premise = f"Title: {meta['show']}\n" + "\n".join(f"Ep {e.number}: {e.logline}" for e in first)
    rng = random.Random(7)
    clips, key = [], {}
    for d in promos[: a.top]:
        v = variants(d, eps)
        if not v:
            continue
        for kind, seg in v.items():
            cid = f"C{rng.randrange(10**5):05d}"
            key[cid] = (d["id"], kind)
            clips.append((cid, Promo(cid, "single", "", [seg])))
    rng.shuffle(clips)

    llm = LLM(LLMConfig.from_env(prefix="JUDGE_", model=a.model))   # the same judge the run used
    scores = {}
    for i in range(0, len(clips), 12):
        batch = clips[i:i + 12]
        text = "\n\n".join(f"=== CLIP {cid} ({p.duration:.0f}s) ===\n{clip_script(p, eps)}" for cid, p in batch)
        data, _ = llm.chat_json(P.JUDGE_SYSTEM, P.JUDGE_USER.format(clips=text, premise=premise))
        for r in data.get("ratings", []):
            if r.get("id") in key:
                scores[key[r["id"]]] = r

    rows, hook_ok, cliff_ok, n = [], 0, 0, 0
    for pid in dict.fromkeys(k[0] for k in key.values()):
        o, s, e = (scores.get((pid, k)) for k in ("original", "slow-start", "early-cut"))
        if not (o and s and e):
            continue
        n += 1
        hook_ok += o["hook"] > s["hook"]
        cliff_ok += o["cliffhanger"] > e["cliffhanger"]
        rows.append({"promo": pid, "original": [o["hook"], o["cliffhanger"]],
                     "slow_start_hook": s["hook"], "early_cut_cliffhanger": e["cliffhanger"]})
        print(f"{pid:8s} original hook {o['hook']} cliff {o['cliffhanger']} | slow-start hook {s['hook']} "
              f"| early-cut cliff {e['cliffhanger']}")
    print(f"\nOriginal hook beats slow-start:       {hook_ok}/{n}")
    print(f"Original cliffhanger beats early-cut: {cliff_ok}/{n}")
    (run / "judge_check.json").write_text(json.dumps({"model": llm.cfg.model, "pairs": n, "hook_wins": hook_ok,
                                                      "cliff_wins": cliff_ok, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
