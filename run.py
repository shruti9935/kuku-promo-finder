"""Find promos in a folder of screenplays.

  python run.py --input "data/<show folder>"                     # whole show
  python run.py --input "data/<show folder>" --episodes 1-5      # quick test
  python run.py --input ... --provider groq --model llama-3.3-70b-versatile
"""
import argparse
import os
import sys
from pathlib import Path

from promo_finder.llm import LLM, LLMConfig, PRESETS
from promo_finder.pipeline import Run


def load_dotenv(path=".env"):
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def parse_range(s: str | None) -> list[int] | None:
    if not s:
        return None
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    load_dotenv()
    ap = argparse.ArgumentParser(description="Find ranked promo clips in microdrama screenplays.")
    ap.add_argument("--input", required=True, help="folder of episode .txt screenplays")
    ap.add_argument("--episodes", help="e.g. 1-5 or 1,4,9 (default: all)")
    ap.add_argument("--provider", choices=list(PRESETS) + ["custom"], help="default: LLM_PROVIDER or gemini")
    ap.add_argument("--model", help="override the preset's model")
    ap.add_argument("--base-url", help="any OpenAI-compatible endpoint")
    ap.add_argument("--judge-provider", choices=list(PRESETS) + ["custom"],
                    help="judge provider (default: JUDGE_PROVIDER, else same as finder)")
    ap.add_argument("--judge-model", help="judge model (default: JUDGE_MODEL, else the finder's model)")
    ap.add_argument("--same-judge", action="store_true", help="let the finder's model judge its own picks")
    ap.add_argument("--no-extras", action="store_true",
                    help="skip the experimental extras: flash-forward hooks, drop-off curves, ad packages (saves ~5 AI calls)")
    ap.add_argument("--max-promos", type=int, default=3, help="candidates per episode")
    ap.add_argument("--max-montages", type=int, default=5, help="cross-episode montages (0 = off)")
    ap.add_argument("--batch", type=int, default=5, help="episodes per AI call (fewer calls fit free tiers)")
    ap.add_argument("--judge-batch", type=int, default=20, help="clips per judge call")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached replies (fresh model calls)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--show", default="", help="show name (default: folder name)")
    a = ap.parse_args()

    cfg = LLMConfig.from_env(provider=a.provider, model=a.model, base_url=a.base_url,
                             temperature=a.temperature)
    cfg.use_cache = not a.no_cache
    judge = None
    if not a.same_judge and (a.judge_model or a.judge_provider or os.getenv("JUDGE_MODEL") or os.getenv("JUDGE_PROVIDER")):
        jp = a.judge_provider or os.getenv("JUDGE_PROVIDER") or cfg.provider
        jcfg = LLMConfig.from_env(prefix="JUDGE_", provider=jp, model=a.judge_model, temperature=a.temperature)
        jcfg.use_cache = cfg.use_cache
        judge = LLM(jcfg)
    run = Run(a.input, LLM(cfg), out_root=a.out, show=a.show, episodes=parse_range(a.episodes),
              max_promos=a.max_promos, max_montages=a.max_montages,
              batch_size=a.batch, judge_batch=a.judge_batch, judge_llm=judge,
              extras=not a.no_extras)
    print(f"Finder: {cfg.model} ({cfg.provider}) · judge: {run.judge_llm.cfg.model} · {len(run.selected)} episodes")
    out = run.execute()
    if run.aborted:
        print(f"! Stopped early: {run.aborted}")
    print(f"\nDone. Outputs in {out}")
    print((out / "report.md").read_text(encoding="utf-8").split("## All promos")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
