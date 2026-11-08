"""Experimental extras, run after the main pipeline has ranked the promos.

  flash_forward  open a promo on a 1-4 s flash of its most shocking moment, then cut back to the
                 scene's start (a common microdrama ad trick). Kept only if the judge scores it higher.
  retention      a predicted "still watching" curve per promo: of 100 scrollers, how many are left
                 after every 3 s. Shown as a chart in the app and as a column in the CSV.
  ad_package     what marketing needs to post a promo: on-screen overlay text (2 versions to A/B test),
                 caption, hashtags, a Hinglish overlay, the best-fit audience, and an EDL cut list
                 that Premiere / DaVinci Resolve / Final Cut can import.

All three are judgment calls by the AI, so they are labelled as predictions, never as facts.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import prompts as P
from .llm import QuotaExhausted
from .parse import fmt_ts
from .pipeline import (MAX_TEASER, SPOKEN, Promo, Segment, check_rules, clean_id, clip_script, lines_in, score)

RETENTION_STEP = 3  # seconds per window


def _call(run, llm, system, user, name):
    """One AI call with the run's usual quota and error handling. Returns parsed JSON or None."""
    if run.aborted:
        return None
    try:
        data, raw = llm.chat_json(system, user)
    except QuotaExhausted as e:
        run._stop(name, e)
        return None
    except Exception as e:
        run.errors.append(f"{name}: {e}")
        return None
    run._raw(f"{name}.json", raw)
    return data


# ---------- 1. flash-forward hooks ----------

def flash_forward(run, promos: list[Promo], top: int = 15) -> list[Promo]:
    """Try a teaser-first version of the best single-scene promos. Returns the variants that the
    judge scored higher than their originals (they then compete in the normal ranking)."""
    cands = [p for p in sorted(promos, key=score, reverse=True) if p.kind == "single" and p.judge][:top]
    if not cands:
        return []
    run.log(f"[flash-forward] looking for teasers in the top {len(cands)}")
    blocks, by_id = [], {p.id: p for p in cands}
    for p in cands:
        seg = p.segments[0]
        rows = [f"=== CLIP {p.id} ({p.duration:.0f}s) ==="]
        for l in lines_in(run.eps[seg.episode], seg):
            if l.kind in SPOKEN:
                who = l.speaker if l.kind == "DIALOGUE" else l.kind.lower()
                rows.append(f"E{seg.episode:02d}-{l.lid} {l.start - seg.start:5.1f}s ({l.end - l.start:.1f}s long) {who}: {l.text[:150]}")
        blocks.append("\n".join(rows))
    data = _call(run, run.llm, P.FINDER_SYSTEM, P.TEASER_USER.format(clips="\n\n".join(blocks)), "flash_forward")
    variants = []
    for t in (data or {}).get("teasers", []) or []:
        orig = by_id.get(clean_id(t.get("id", "")))
        lid = str(t.get("line", "")).strip()
        if not orig or not lid or lid.lower() == "none":
            continue
        seg = orig.segments[0]
        ep = run.eps[seg.episode]
        line = ep.line(lid.split("-")[-1])
        if not line or not (seg.start + 2.0 <= line.start < seg.end) or line.end - line.start > MAX_TEASER:
            continue                       # the teaser must come from later in the clip, and be short
        if line.end >= seg.end - 0.01:
            continue                       # never flash the cliffhanger itself
        teaser = Segment(seg.episode, line.start, line.end, line.lid, line.lid)
        body = seg
        if orig.duration + teaser.duration > 60:   # no room for the teaser
            continue
        v = Promo(id=f"{orig.id}f", kind="flash-forward", title=orig.title, segments=[teaser, body],
                  hook=f"Flash-forward: {t.get('why', '')}".strip(), cliffhanger=orig.cliffhanger, why=orig.why,
                  context_needed=orig.context_needed)
        if check_rules(v, run.eps):
            continue
        variants.append(v)
    if not variants:
        return []
    run.judge(variants, "flash_forward")
    kept = [v for v in variants if v.judge and score(v) > score(by_id[v.id[:-1]])]
    run.flash_stats = {"tried": len(variants), "kept": len(kept)}
    for v in kept:
        v.flags.append(f"beats the plain version {by_id[v.id[:-1]].id} ({score(v)} vs {score(by_id[v.id[:-1]])})")
    return kept


# ---------- 2. predicted drop-off curve ----------

def retention(run, ranked: list[Promo], top: int = 20, batch: int = 10):
    """Adds judge['retention'] (still watching out of 100, every 3 s) and judge['biggest_drop']."""
    cands = [p for p in ranked[:top] if p.judge]
    if not cands:
        return
    run.log(f"[retention] predicting drop-off for the top {len(cands)}")
    for i in range(0, len(cands), batch):
        part = cands[i:i + batch]
        clips = "\n\n".join(f"=== CLIP {p.id} ({p.duration:.0f}s) ===\n{clip_script(p, run.eps)}" for p in part)
        data = _call(run, run.judge_llm, P.JUDGE_SYSTEM, P.RETENTION_USER.format(clips=clips), f"retention_{i // batch + 1}")
        by_id = {clean_id(c.get("id", "")): c for c in (data or {}).get("curves", []) or []}
        for p in part:
            c = by_id.get(p.id)
            if not c:
                continue
            curve = clean_curve(c.get("still_watching"), p.duration)
            if curve:
                p.judge["retention"] = curve
                p.judge["watch_through"] = curve[-1]
                p.judge["biggest_drop"] = str(c.get("biggest_drop", ""))[:160]


def clean_curve(values, duration: float) -> list[int]:
    """Numbers only, 0-100, never rising, one value per 3 s window (padded or cut to fit)."""
    out, prev = [], 100
    for v in values or []:
        try:
            v = int(round(float(v)))
        except (TypeError, ValueError):
            continue
        v = max(0, min(prev, v))
        out.append(v)
        prev = v
    n = max(1, int(-(-duration // RETENTION_STEP)))   # ceil
    if not out:
        return []
    out = out[:n] + [out[-1]] * (n - len(out))
    return out


# ---------- 3. ad package + editor cut lists ----------

def ad_package(run, promos: list[Promo]) -> dict:
    """Overlay text, caption, hashtags, Hinglish overlay and audience for the given promos."""
    if not promos:
        return {}
    run.log(f"[ad copy] writing ad packages for {len(promos)} promos")
    clips = "\n\n".join(f"=== PROMO {p.id}: {p.title} ({p.duration:.0f}s) ===\nHook: {p.hook}\nCliffhanger: {p.cliffhanger}\n"
                        f"{clip_script(p, run.eps)}" for p in promos)
    data = _call(run, run.llm, P.FINDER_SYSTEM, P.ADCOPY_USER.format(show=run.show, clips=clips), "ad_copy")
    out = {}
    for a in (data or {}).get("ads", []) or []:
        pid = clean_id(a.get("id", ""))
        if pid in {p.id for p in promos}:
            out[pid] = {k: a.get(k) for k in ("overlay", "overlay_b", "caption", "hashtags", "hinglish", "audience")}
    return out


def _tc(sec: float, fps: int = 30) -> str:
    f = int(round(sec * fps))
    return f"{f // (3600 * fps):02d}:{f // (60 * fps) % 60:02d}:{f // fps % 60:02d}:{f % fps:02d}"


def edl(p: Promo, fps: int = 30) -> str:
    """CMX 3600 edit decision list. Each episode is its own reel (EP012), so an editor relinks the
    reels to the episode video files and the whole promo is cut automatically."""
    rows = [f"TITLE: {p.id} {p.title}"[:70], "FCM: NON-DROP FRAME", ""]
    rec = 0.0
    for i, s in enumerate(p.segments, 1):
        rows.append(f"{i:03d}  EP{s.episode:03d}    V     C        {_tc(s.start, fps)} {_tc(s.end, fps)} "
                    f"{_tc(rec, fps)} {_tc(rec + s.duration, fps)}")
        rows.append(f"* FROM CLIP NAME: episode {s.episode} {fmt_ts(s.start)}-{fmt_ts(s.end)} ({s.start_line}-{s.end_line})")
        rows.append(f"{i:03d}  EP{s.episode:03d}    A     C        {_tc(s.start, fps)} {_tc(s.end, fps)} "
                    f"{_tc(rec, fps)} {_tc(rec + s.duration, fps)}")
        rec += s.duration
    return "\n".join(rows) + "\n"


def write_extras(run, ranked: list[Promo], ads: dict):
    d = run.dir / "edl"
    d.mkdir(exist_ok=True)
    for p in ranked[:20]:
        (d / f"{p.id}.edl").write_text(edl(p), encoding="utf-8")
    (run.dir / "ads.json").write_text(json.dumps(ads, indent=2, ensure_ascii=False), encoding="utf-8")


def run_extras(run, ranked: list[Promo]) -> list[Promo]:
    """Called by Run.execute() after ranking. Returns the final ranked list."""
    retention(run, ranked)
    shortlist = run.shortlist(ranked, n=10)
    run.ads = ad_package(run, shortlist)
    write_extras(run, ranked, run.ads)
    return ranked
