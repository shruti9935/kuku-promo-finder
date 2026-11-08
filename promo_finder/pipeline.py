"""Promo finder pipeline.

  parse (code) -> find candidates per episode (AI) -> resolve + check rules (code)
  -> judge each clip cold, as a stranger (AI) -> montages across episodes (AI) -> rank (code)

The AI only ever names line IDs. Timestamps, durations, cut points and rule checks are
computed in code, so a promo can never point at a time that isn't in the script.
"""
from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import prompts as P
from .llm import LLM, QuotaExhausted
from .parse import Episode, Line, fmt_ts, load_episodes, render_for_llm

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_WINDOW = 3.0
MAX_SEGMENTS = 3              # more cuts than this reads as choppy in a 20-60 s ad
MAX_TEASER = 4.0              # a flash-forward teaser is a single 1-4 s line
MIN_SEGMENT = 3.0             # a piece shorter than this is a jump cut, not a scene
SPOKEN = ("DIALOGUE", "ACTION", "SFX")

# Ranking weights. Simple and visible on purpose so an editor can argue with them.
W_HOOK, W_CLIFF, W_STANDALONE = 0.4, 0.3, 0.3
PENALTY_NO_HOOK_IN_3S = 0.5   # code found no dialogue/action/sfx starting in the first 3 s
PENALTY_SPOILER = 1.0         # spoilers go to a separate "hold for later" list too
PENALTY_WEAK_CLIFF = 0.5      # judge cliffhanger <= 2: the cut lands on a resolved beat
WEAK_CLIFF = 2
EARLY_EPISODE_BONUS = 0.25    # promo comes from eps 1-10: new viewers land near Episode 1
EARLY_EPISODES = 10
HEAD_TO_HEAD = 20             # top clips re-ordered by one side-by-side comparison


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    start_line: str
    end_line: str

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def to_dict(self) -> dict:
        return {"episode": self.episode, "start": fmt_ts(self.start), "end": fmt_ts(self.end),
                "start_sec": self.start, "end_sec": self.end, "duration": self.duration,
                "start_line": self.start_line, "end_line": self.end_line}


@dataclass
class Promo:
    id: str
    kind: str                      # single | multi-segment | montage
    title: str
    segments: list[Segment]
    hook: str = ""
    cliffhanger: str = ""
    why: str = ""
    context_needed: str = ""
    flags: list[str] = field(default_factory=list)
    hook_in_3s: bool = False
    opening_line: str = ""
    closing_line: str = ""
    judge: dict = field(default_factory=dict)
    score: float = 0.0
    duplicate_of: str = ""

    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)

    @property
    def episodes(self) -> list[int]:
        return sorted({s.episode for s in self.segments})

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "title": self.title, "episodes": self.episodes,
                "duration": self.duration, "segments": [s.to_dict() for s in self.segments],
                "hook": self.hook, "cliffhanger": self.cliffhanger, "why": self.why,
                "context_needed": self.context_needed, "opening_line": self.opening_line,
                "closing_line": self.closing_line, "hook_in_3s": self.hook_in_3s,
                "judge": self.judge, "score": self.score, "flags": self.flags,
                "duplicate_of": self.duplicate_of}


# ---------- code: resolve line IDs to clean cut points ----------

def resolve_segment(ep: Episode, start_lid: str, end_lid: str) -> tuple[Segment | None, str]:
    a, b = ep.line(start_lid), ep.line(end_lid)
    if a is None or b is None:
        return None, f"unknown line id ({start_lid}..{end_lid}) in ep {ep.number}"
    if b.idx < a.idx:
        a, b = b, a
    start, end = a.start, max(l.end for l in ep.lines[a.idx - 1:b.idx])
    # never start in the middle of someone's sentence
    for l in ep.lines:
        if l.kind == "DIALOGUE" and l.start < start < l.end:
            start = l.start
    # never cut in the middle of a line, a sound or an action
    changed = True
    while changed:
        changed = False
        for l in ep.lines:
            if l.kind in SPOKEN and start <= l.start < end < l.end:
                end, changed = l.end, True
    # finish the camera shot if it ends within 1.5 s (cleaner cut for the editor)
    for l in ep.lines:
        if l.kind == "CAMERA" and l.start < end < l.end <= end + 1.5:
            end = l.end
    return Segment(ep.number, round(start, 2), round(end, 2), a.lid, b.lid), ""


def lines_in(ep: Episode, seg: Segment) -> list[Line]:
    return [l for l in ep.lines if l.start >= seg.start - 0.01 and l.start < seg.end]


def trim_to_fit(promo: Promo, eps: dict[int, Episode]) -> bool:
    """A clip that runs past 60 s is cut back to the last spoken line that still fits,
    instead of being thrown away. The judge then decides if the new ending still hooks."""
    before = promo.duration
    while promo.duration > MAX_LEN and promo.segments:
        seg = promo.segments[-1]
        room = MAX_LEN - (promo.duration - seg.duration)
        ep = eps[seg.episode]
        fits = [l for l in lines_in(ep, seg) if l.kind in SPOKEN and l.end <= seg.start + room and l.end > seg.start]
        if not fits:
            promo.segments.pop()   # drop the whole last piece and try again
            continue
        new, _ = resolve_segment(ep, seg.start_line, fits[-1].lid)
        if not new or new.duration >= seg.duration:
            return False
        promo.segments[-1] = new
    if promo.segments and promo.duration <= MAX_LEN:
        promo.flags.append(f"trimmed from {before:.0f}s to fit 60s")
        return True
    return False


def check_rules(promo: Promo, eps: dict[int, Episode]) -> str:
    """Fill in rule flags. Returns a rejection reason, or '' if the promo is usable."""
    segs = promo.segments
    if not segs:
        return "no segments"
    ff = promo.kind == "flash-forward"          # a short teaser first, then the scene in story order
    if len(segs) > MAX_SEGMENTS + (1 if promo.kind in ("montage", "flash-forward") else 0):
        return f"too many cuts ({len(segs)} segments)"
    if ff and not (0 < segs[0].duration <= MAX_TEASER):
        return f"teaser longer than {MAX_TEASER:.0f}s"
    body = segs[1:] if ff else segs
    if len(segs) > 1 and any(s.duration < MIN_SEGMENT for s in body):
        return f"segment shorter than {MIN_SEGMENT:.0f}s (jump cut)"
    for x, y in zip(body, body[1:]):
        if (y.episode, y.start) < (x.episode, x.start):
            return "segments out of story order"
        if y.episode == x.episode and y.start < x.end:
            return "segments overlap"
    d = promo.duration
    if d < MIN_LEN:
        return f"too short ({d:.1f}s)"
    if d > MAX_LEN:
        if not trim_to_fit(promo, eps):
            return f"too long ({d:.1f}s)"
        segs = promo.segments
        if any(x.duration < MIN_SEGMENT for x in (segs[1:] if ff else segs)):
            return "too long, and trimming left a jump cut"
        if promo.duration < MIN_LEN:
            return f"too short after trimming ({promo.duration:.1f}s)"
    first = segs[0]
    opening = [l for l in lines_in(eps[first.episode], first) if l.kind in SPOKEN and l.start < first.start + HOOK_WINDOW]
    promo.hook_in_3s = bool(opening)
    if not opening:
        promo.flags.append("nothing said or done in first 3s")
    dlg = [l for l in lines_in(eps[first.episode], first) if l.kind == "DIALOGUE"]
    promo.opening_line = _short(opening[0] if opening else (dlg[0] if dlg else None))
    last = segs[-1]
    tail = [l for l in lines_in(eps[last.episode], last) if l.kind in SPOKEN]
    promo.closing_line = _short(tail[-1] if tail else None)
    if ff:
        promo.flags.append("flash-forward: opens on a teaser, then cuts back")
    elif len(segs) > 1:
        promo.flags.append(f"{len(segs)} segments: needs an edit")
    return ""


def _short(l: Line | None) -> str:
    if l is None:
        return ""
    who = l.speaker if l.kind == "DIALOGUE" else l.kind.lower()
    return f"{who}: {l.text}"


def clip_script(promo: Promo, eps: dict[int, Episode], camera_chars: int = 100) -> str:
    """The clip exactly as a viewer meets it, with times relative to the clip's first frame."""
    out, offset = [], 0.0
    for i, seg in enumerate(promo.segments):
        if i:
            out.append("--- CUT ---")
        for l in lines_in(eps[seg.episode], seg):
            t = offset + (l.start - seg.start)
            if l.kind == "DIALOGUE":
                out.append(f"{t:5.1f}s {l.speaker}: {l.text}")
            elif l.kind == "CAMERA":
                txt = l.text if len(l.text) <= camera_chars else l.text[:camera_chars] + "…"
                out.append(f"{t:5.1f}s (camera) {txt}")
            else:
                out.append(f"{t:5.1f}s {l.kind.lower()}: {l.text}")
        offset += seg.duration
    out.append(f"{offset:5.1f}s [CLIP ENDS]")
    return "\n".join(out)


def clean_id(x) -> str:
    """Models sometimes echo the label: 'CLIP E03-P1' or '[E03-P1]' -> 'E03-P1'."""
    return re.sub(r"^\W*(clip\s*)?", "", str(x).strip(), flags=re.I).strip(" []()#:")


def overlap_ratio(p: Promo, q: Promo) -> float:
    best = 0.0
    for a in p.segments:
        for b in q.segments:
            if a.episode != b.episode:
                continue
            inter = min(a.end, b.end) - max(a.start, b.start)
            if inter > 0:
                best = max(best, inter / min(a.duration, b.duration))
    return best


def score(p: Promo) -> float:
    j = p.judge
    if not j:
        return 0.0
    s = W_HOOK * j.get("hook", 1) + W_CLIFF * j.get("cliffhanger", 1) + W_STANDALONE * j.get("standalone", 1)
    if not p.hook_in_3s:
        s -= PENALTY_NO_HOOK_IN_3S
    if j.get("spoiler"):
        s -= PENALTY_SPOILER
    if j.get("cliffhanger", 1) <= WEAK_CLIFF:
        s -= PENALTY_WEAK_CLIFF
    if min(p.episodes) <= EARLY_EPISODES:
        s += EARLY_EPISODE_BONUS
    return round(s, 2)


# ---------- the run ----------

class Run:
    def __init__(self, input_dir: str, llm: LLM, out_root: str = "outputs", show: str = "",
                 episodes: list[int] | None = None, max_promos: int = 3, max_montages: int = 5,
                 batch_size: int = 5, judge_batch: int = 20, log: Callable[[str], None] = print,
                 judge_llm: LLM | None = None, extras: bool = True):
        self.input_dir = Path(input_dir).resolve()
        self.all_eps = load_episodes(input_dir)
        if not self.all_eps:
            raise ValueError(f"No screenplay .txt files found in {input_dir}")
        self.eps = {e.number: e for e in self.all_eps}
        self.selected = [e for e in self.all_eps if not episodes or e.number in episodes]
        self.llm, self.log = llm, log
        self.judge_llm = judge_llm or llm   # a different model grades the finder's work
        self.show = show or Path(input_dir).name
        self.max_promos, self.max_montages = max_promos, max_montages
        self.batch_size, self.judge_batch = max(1, batch_size), max(1, judge_batch)
        self.aborted = ""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(out_root) / f"run-{stamp}"
        (self.dir / "raw").mkdir(parents=True, exist_ok=True)
        self.promos: list[Promo] = []
        self.rejected: list[dict] = []
        self.moments: list[dict] = []
        self.errors: list[str] = []
        self.recut_log: list[dict] = []
        self.extras = extras                # flash-forward hooks, drop-off curves, ad packages (extras.py)
        self.flash_stats = {"tried": 0, "kept": 0}
        self.ads: dict = {}
        self.recut_stats = {"weak_endings": 0, "recut_tried": 0, "recut_kept": 0}

    def _premise(self) -> str:
        first = "\n".join(f"Ep {e.number}: {e.logline}" for e in self.all_eps[:3])
        return f"Title: {self.show}\n{first}"

    def _story(self, upto: int) -> str:
        return "\n".join(f"Ep {e.number}: {e.logline}" for e in self.all_eps if e.number <= upto)

    def _raw(self, name: str, text: str):
        (self.dir / "raw" / name).write_text(text, encoding="utf-8")

    def _episode_of(self, item: dict, lid: str, batch: list[Episode]) -> Episode | None:
        """Which episode a model-picked line belongs to: from 'E03-L012', else the item's 'episode'."""
        m = re.match(r"\s*E\s*(\d+)", str(lid), re.I)
        n = int(m.group(1)) if m else item.get("episode")
        try:
            n = int(n)
        except (TypeError, ValueError):
            return batch[0] if len(batch) == 1 else None
        return self.eps[n] if any(e.number == n for e in batch) else None

    def _stop(self, where: str, e: Exception):
        self.aborted = str(e)
        self.errors.append(f"{where}: {e}")
        self.log(f"  ! quota used up, no more AI calls this run: {e}")

    def find(self):
        batches = [self.selected[i:i + self.batch_size] for i in range(0, len(self.selected), self.batch_size)]
        counters: dict[int, int] = {}
        for n, batch in enumerate(batches, 1):
            nums = [e.number for e in batch]
            if self.aborted:
                self.errors.append(f"find skipped for episodes {nums[0]}-{nums[-1]} (quota)")
                continue
            self.log(f"[find {n}/{len(batches)}] episodes {nums[0]}-{nums[-1]}")
            blocks = "\n\n".join(P.EPISODE_BLOCK.format(ep=e.number, logline=e.logline, duration=fmt_ts(e.duration),
                                                       script=render_for_llm(e)) for e in batch)
            user = P.FINDER_USER.format(show=self.show, story=self._story(nums[-1]), episodes=blocks,
                                        max_promos=self.max_promos)
            try:
                data, raw = self.llm.chat_json(P.FINDER_SYSTEM, user)
            except QuotaExhausted as e:
                self._stop(f"find episodes {nums[0]}-{nums[-1]}", e)
                continue
            except Exception as e:
                self.errors.append(f"find episodes {nums[0]}-{nums[-1]}: {e}")
                self.log(f"  ! failed: {e}")
                continue
            self._raw(f"find_ep{nums[0]:03d}-{nums[-1]:03d}.json", raw)
            for c in data.get("promos", []) or []:
                raw_segs = c.get("segments", []) or []
                ep = self._episode_of(c, raw_segs[0].get("start_line", "") if raw_segs else "", batch)
                if ep is None:
                    self.rejected.append({"id": f"B{n}-{len(self.rejected) + 1}", "reason": "episode not in this batch",
                                          "candidate": c})
                    continue
                counters[ep.number] = counters.get(ep.number, 0) + 1
                pid = f"E{ep.number:02d}-P{counters[ep.number]}"
                segs, err = [], ""
                for s in raw_segs:
                    if self._episode_of(c, s.get("end_line", ""), batch) is not ep:
                        err = "segment spans two episodes"
                        break
                    seg, err = resolve_segment(ep, s.get("start_line", ""), s.get("end_line", ""))
                    if not seg:
                        break
                    segs.append(seg)
                promo = Promo(id=pid, kind="single" if len(segs) == 1 else "multi-segment",
                              title=c.get("title", ""), segments=segs, hook=c.get("hook", ""),
                              cliffhanger=c.get("cliffhanger", ""), why=c.get("why", ""),
                              context_needed=c.get("context_needed", ""))
                reason = err or check_rules(promo, self.eps)
                if reason:
                    self.rejected.append({"id": pid, "reason": reason, "candidate": c,
                                          "resolved_duration": promo.duration if segs else None})
                else:
                    self.promos.append(promo)
            for m in data.get("key_moments", []) or []:
                ep = self._episode_of(m, m.get("start_line", ""), batch)
                if ep is None:
                    continue
                seg, err = resolve_segment(ep, m.get("start_line", ""), m.get("end_line", ""))
                if seg and MIN_SEGMENT <= seg.duration <= 25.0:
                    self.moments.append({"id": f"M{len(self.moments) + 1:03d}", "segment": seg,
                                         "type": m.get("type", ""), "what": m.get("what", "")})

    def judge(self, promos: list[Promo], tag: str):
        size = self.judge_batch
        for i in range(0, len(promos), size):
            batch = promos[i:i + size]
            if self.aborted:
                for p in batch:
                    p.flags.append("not judged: quota used up")
                continue
            self.log(f"[judge {tag}] clips {i + 1}-{i + len(batch)} of {len(promos)}")
            clips = "\n\n".join(f"=== CLIP {p.id} ({p.duration:.0f}s) ===\n{clip_script(p, self.eps)}" for p in batch)
            try:
                data, raw = self.judge_llm.chat_json(P.JUDGE_SYSTEM, P.JUDGE_USER.format(clips=clips, premise=self._premise()))
            except QuotaExhausted as e:
                self._stop(f"judge {tag}", e)
                for p in batch:
                    p.flags.append("not judged: quota used up")
                continue
            except Exception as e:
                self.errors.append(f"judge {tag}: {e}")
                self.log(f"  ! judge failed: {e}")
                continue
            self._raw(f"judge_{tag}_{i // size + 1}.json", raw)
            by_id = {clean_id(r.get("id", "")): r for r in data.get("ratings", []) or []}
            for p in batch:
                r = by_id.get(p.id)
                if r:
                    p.judge = {k: r.get(k) for k in ("hook", "cliffhanger", "standalone", "spoiler", "critique")}
                    for k in ("hook", "cliffhanger", "standalone"):
                        try:
                            p.judge[k] = max(1, min(5, int(p.judge[k])))
                        except Exception:
                            p.judge[k] = 1
                    p.judge["spoiler"] = bool(p.judge.get("spoiler"))
                else:
                    p.flags.append("judge skipped this clip")

    def montages(self) -> list[Promo]:
        if len(self.moments) < 2 or self.max_montages <= 0 or self.aborted:
            return []
        self.log(f"[montage] building from {len(self.moments)} key moments")
        lines = []
        for m in self.moments:
            s, ep = m["segment"], self.eps[m["segment"].episode]
            ls = [l for l in lines_in(ep, s) if l.kind in SPOKEN]
            first, last = (_short(ls[0]), _short(ls[-1])) if ls else ("", "")
            lines.append(f"{m['id']} | ep {s.episode} | {s.duration:.0f}s | {m['type']} | {m['what']} | "
                         f"first: {first[:120]} | last: {last[:120]}")
        maxep = max(e.number for e in self.selected)
        user = P.MONTAGE_USER.format(show=self.show, story=self._story(maxep), moments="\n".join(lines),
                                     max_montages=self.max_montages)
        try:
            data, raw = self.llm.chat_json(P.MONTAGE_SYSTEM, user)
        except QuotaExhausted as e:
            self._stop("montage", e)
            return []
        except Exception as e:
            self.errors.append(f"montage: {e}")
            return []
        self._raw("montages.json", raw)
        by_id = {m["id"]: m for m in self.moments}
        out = []
        for k, c in enumerate(data.get("montages", []) or [], 1):
            pid = f"X-{k}"
            ms = [by_id[i] for i in c.get("moment_ids", []) if i in by_id]
            segs = [m["segment"] for m in ms]
            promo = Promo(id=pid, kind="montage", title=c.get("title", ""), segments=segs,
                          hook=c.get("hook", ""), cliffhanger=c.get("cliffhanger", ""), why=c.get("why", ""))
            if len(segs) < 2:
                self.rejected.append({"id": pid, "reason": "fewer than 2 valid moments", "candidate": c})
                continue
            ordered = sorted(segs, key=lambda s: (s.episode, s.start))
            if ordered != segs:
                promo.segments = ordered
                promo.flags.append("moments reordered into story order")
            reason = check_rules(promo, self.eps)
            if reason:
                self.rejected.append({"id": pid, "reason": reason, "candidate": c})
            else:
                out.append(promo)
        return out

    def rank(self, promos: list[Promo]) -> list[Promo]:
        for p in promos:
            p.score = score(p)
        promos.sort(key=lambda p: (-p.score, -(p.judge.get("hook") or 0), p.duration))
        kept: list[Promo] = []
        for p in promos:
            # montages are a different kind of ad, so only compare like with like
            dup = next((k for k in kept if (k.kind == "montage") == (p.kind == "montage")
                        and overlap_ratio(p, k) > 0.5), None)
            if dup:
                p.duplicate_of = dup.id
            else:
                kept.append(p)
        return kept

    def recut(self, promos: list[Promo]):
        """The most common miss: the finder runs a line or two past the cliffhanger into a resolution.
        For clips the judge says end on a resolved beat, an editor call picks an earlier end line,
        code rebuilds the cut, the judge re-scores it, and the new cut is kept only if it scores better."""
        weak = [p for p in promos if p.judge and p.judge.get("cliffhanger", 5) <= WEAK_CLIFF]
        self.recut_stats = {"weak_endings": len(weak), "recut_tried": 0, "recut_kept": 0}
        if not weak or self.aborted:
            return
        self.log(f"[re-cut] {len(weak)} clips end on a resolved beat")
        listing, by_id = [], {p.id: p for p in weak}
        for p in weak:
            seg, offset = p.segments[-1], p.duration - p.segments[-1].duration
            rows = [f"=== CLIP {p.id} ({p.duration:.0f}s) ==="]
            for l in lines_in(self.eps[seg.episode], seg):
                if l.kind in SPOKEN:
                    who = l.speaker if l.kind == "DIALOGUE" else l.kind.lower()
                    rows.append(f"E{seg.episode:02d}-{l.lid} ends {offset + l.end - seg.start:5.1f}s {who}: {l.text[:160]}")
            listing.append("\n".join(rows))
        new: list[Promo] = []
        for i in range(0, len(listing), self.judge_batch):
            try:
                data, raw = self.llm.chat_json(P.RECUT_SYSTEM, P.RECUT_USER.format(clips="\n\n".join(listing[i:i + self.judge_batch])))
            except QuotaExhausted as e:
                self._stop("re-cut", e)
                return
            except Exception as e:
                self.errors.append(f"re-cut: {e}")
                continue
            self._raw(f"recut_{i // self.judge_batch + 1}.json", raw)
            for r in data.get("recuts", []) or []:
                old = by_id.get(clean_id(r.get("id", "")))
                end = str(r.get("end_line", "")).strip()
                if not old or not end or end.lower() == "keep":
                    continue
                seg = old.segments[-1]
                ep = self.eps[seg.episode]
                if self._episode_of({}, end, [ep]) is not ep:
                    continue
                cut, err = resolve_segment(ep, seg.start_line, end)
                if err or not cut or cut.start != seg.start or cut.end >= seg.end:
                    continue
                q = Promo(id=f"{old.id}r", kind=old.kind, title=old.title, segments=old.segments[:-1] + [cut],
                          hook=old.hook, cliffhanger=r.get("why", "") or old.cliffhanger, why=old.why,
                          context_needed=old.context_needed)
                if check_rules(q, self.eps):
                    continue
                q.flags.append(f"re-cut: ends {seg.end - cut.end:.1f}s earlier than the finder's pick ({old.id})")
                new.append(q)
        self.recut_stats["recut_tried"] = len(new)
        if not new:
            return
        self.judge(new, "recuts")
        for q in new:
            old = by_id[q.id[:-1]]
            if q.judge and score(q) > score(old):
                promos[promos.index(old)] = q
                self.recut_stats["recut_kept"] += 1
                self.recut_log.append({"original": old.to_dict(), "recut": q.to_dict()})

    def head_to_head(self, ranked: list[Promo]) -> list[Promo]:
        """Absolute 1-5 scores bunch up at the top (many clips get 5/5/5). One side-by-side call
        orders the top clips against each other; their scores stay as they were."""
        top = [p for p in ranked[:HEAD_TO_HEAD] if p.judge]
        if len(top) < 3 or self.aborted:
            return ranked
        self.log(f"[head-to-head] ordering the top {len(top)}")
        clips = "\n\n".join(f"=== CLIP {p.id} ({p.duration:.0f}s) ===\n{clip_script(p, self.eps)}" for p in top)
        try:
            data, raw = self.judge_llm.chat_json(P.RANK_SYSTEM, P.RANK_USER.format(clips=clips))
        except QuotaExhausted as e:
            self._stop("head-to-head", e)
            return ranked
        except Exception as e:
            self.errors.append(f"head-to-head: {e}")
            return ranked
        self._raw("head_to_head.json", raw)
        by_id = {p.id: p for p in top}
        order = [by_id[i] for i in dict.fromkeys(clean_id(i) for i in data.get("order", []) or []) if i in by_id]
        if len(order) < len(top) // 2:      # the model ignored the task: keep the score order
            self.errors.append("head-to-head: reply did not list the clips")
            return ranked
        order += [p for p in top if p not in order]
        for i, p in enumerate(order, 1):
            p.judge["head_to_head"] = i
        rest = [p for p in ranked if p not in order]
        self.head_to_head_reason = data.get("best_reason", "")
        return order + rest

    def execute(self) -> Path:
        t0 = time.time()
        self.find()
        self.log(f"[judge] {len(self.promos)} single-episode candidates")
        self.judge(sorted(self.promos, key=lambda p: (p.episodes[0], p.segments[0].start)), "singles")
        self.recut(self.promos)
        if self.extras:
            from . import extras
            self.promos += extras.flash_forward(self, self.promos)
        mont = self.montages()
        if mont:
            self.judge(mont, "montages")
        self.montages_passed = len(mont)
        everything = self.promos + mont
        ranked = self.head_to_head(self.rank(list(everything)))
        if self.extras:
            from . import extras
            ranked = extras.run_extras(self, ranked)
        dups = [p for p in everything if p.duplicate_of]
        self.write(ranked, dups, time.time() - t0)
        return self.dir

    # ---------- outputs ----------

    def write(self, ranked: list[Promo], dups: list[Promo], secs: float):
        d = self.dir
        (d / "promos_ranked.json").write_text(json.dumps(
            [dict(rank=i, **p.to_dict(), ad=self.ads.get(p.id, {})) for i, p in enumerate(ranked, 1)],
            indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "recuts.json").write_text(json.dumps(self.recut_log, indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "rejected.json").write_text(json.dumps(self.rejected, indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "duplicates.json").write_text(json.dumps([p.to_dict() for p in dups], indent=2, ensure_ascii=False), encoding="utf-8")
        with open(d / "promos_ranked.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "id", "kind", "score", "duration_s", "segments", "hook", "cliffhanger", "why",
                        "judge_hook", "judge_cliff", "judge_standalone", "spoiler", "judge_critique", "flags",
                        "predicted_watch_through_pct", "overlay_text", "overlay_text_b", "caption", "hinglish_overlay"])
            for i, p in enumerate(ranked, 1):
                segs = " + ".join(f"Ep{s.episode} {fmt_ts(s.start)}-{fmt_ts(s.end)}" for s in p.segments)
                j = p.judge
                w.writerow([i, p.id, p.kind, p.score, p.duration, segs, p.hook, p.cliffhanger, p.why,
                            j.get("hook"), j.get("cliffhanger"), j.get("standalone"), j.get("spoiler"),
                            j.get("critique"), "; ".join(p.flags), j.get("watch_through"),
                            *(self.ads.get(p.id, {}).get(k) for k in ("overlay", "overlay_b", "caption", "hinglish"))])
        reasons: dict[str, int] = {}
        for r in self.rejected:
            key = r["reason"].split(" (")[0]
            reasons[key] = reasons.get(key, 0) + 1
        meta = {
            "show": self.show, "input_dir": str(self.input_dir), "episodes": [e.number for e in self.selected], "llm": self.llm.cfg.public(),
            "judge_llm": self.judge_llm.cfg.public(), "prompt_version": P.PROMPT_VERSION, "batch_size": self.batch_size,
            "judge_batch": self.judge_batch, "stopped_early": self.aborted, "seconds": round(secs, 1), **self._usage(),
            "candidates_found": len(self.promos) + len([r for r in self.rejected if r["id"].startswith("E")]),
            "passed_rules": len(self.promos), "montages_passed": getattr(self, "montages_passed", 0),
            "rejected_by_reason": reasons, "recut": self.recut_stats, "flash_forward": self.flash_stats,
            "extras": self.extras, "ad_packages": len(self.ads), "duplicates_removed": len(dups), "final_promos": len(ranked),
            "errors": self.errors,
        }
        (d / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (d / "report.md").write_text(self.report(ranked, meta), encoding="utf-8")

    def _usage(self) -> dict:
        llms = [self.llm] + ([self.judge_llm] if self.judge_llm is not self.llm else [])
        per_model: dict[str, int] = {}
        tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        for l in llms:
            for m, n in getattr(l, "model_used", {}).items():
                per_model[m] = per_model.get(m, 0) + n
            for k in tokens:
                tokens[k] += l.usage.get(k, 0)
        return {"llm_calls": sum(l.calls for l in llms), "cache_hits": sum(l.cache_hits for l in llms),
                "calls_per_model": per_model, "tokens": tokens}

    @staticmethod
    def shortlist(ranked: list[Promo], n: int = 5) -> list[Promo]:
        """Best judged, spoiler-free promos, at most one per episode so the first ads differ."""
        out, used = [], set()
        for p in ranked:
            if not p.judge or p.judge.get("spoiler"):
                continue
            key = p.episodes[0] if p.kind != "montage" else "montage"
            if key in used:
                continue
            used.add(key)
            out.append(p)
            if len(out) == n:
                break
        return out

    def report(self, ranked: list[Promo], meta: dict) -> str:
        out = [f"# Promo report: {self.show}", "",
               f"Finder: `{meta['llm']['model']}` · judge: `{meta['judge_llm']['model']}` · prompts {meta['prompt_version']} · "
               f"{len(meta['episodes'])} episodes · {meta['final_promos']} promos after checks and de-duplication.", "",
               "Score = 0.4×hook + 0.3×cliffhanger + 0.3×standalone (judge, 1-5), "
               f"−{PENALTY_NO_HOOK_IN_3S} if nothing happens in the first 3 s, −{PENALTY_SPOILER} if it spoils a twist, "
               f"−{PENALTY_WEAK_CLIFF} if the cliffhanger scores {WEAK_CLIFF} or less, "
               f"+{EARLY_EPISODE_BONUS} if from episodes 1-{EARLY_EPISODES}.", ""]
        if any("head_to_head" in p.judge for p in ranked):
            out += [f"Order: the top {HEAD_TO_HEAD} are ordered by one head-to-head comparison by the judge "
                    "(absolute scores bunch up at 5/5/5); the rest by score.", ""]
        unjudged = [p for p in ranked if not p.judge]
        if meta.get("stopped_early"):
            out += [f"> **Run stopped early:** {meta['stopped_early']} {len(unjudged)} promos were not judged, so "
                    "their order is not meaningful. Re-run after the quota resets; finished calls are cached.", ""]
        def line(i, p):
            j = p.judge
            segs = " + ".join(f"Ep{s.episode} {fmt_ts(s.start)}-{fmt_ts(s.end)}" for s in p.segments)
            return (f"{i}. **{p.title}** ({p.id}, {segs}, {p.duration:.0f}s, score {p.score}). Hook {j.get('hook')}/5, "
                    f"cliffhanger {j.get('cliffhanger')}/5, standalone {j.get('standalone')}/5. {p.why}")
        out += ["## Post these first", "",
                "Best spoiler-free promos, at most one per episode so the first ads are not all the same scene.", ""]
        out += [line(i, p) for i, p in enumerate(self.shortlist(ranked), 1)]
        spoilers = [p for p in ranked if p.judge.get("spoiler")]
        if spoilers:
            out += ["", "## Hold for later (gives away a twist)", "",
                    "Strong clips the judge says spoil the story. Better for re-targeting people who already watch.", ""]
            out += [line(i, p) for i, p in enumerate(spoilers[:5], 1)]
        out += ["", "## Where the tool picks badly", ""] + self.weak_spots(ranked)
        out += self.experimental(ranked) + ["", "## All promos", ""]
        for i, p in enumerate(ranked, 1):
            j = p.judge
            out.append(f"### {i}. {p.title}  ·  score {p.score}  ·  {p.kind}  ·  {p.duration:.0f}s")
            for s in p.segments:
                out.append(f"- Episode {s.episode}: {fmt_ts(s.start)} → {fmt_ts(s.end)} ({s.duration:.1f}s, lines {s.start_line}–{s.end_line})")
            out.append(f"- **Hook:** {p.hook}")
            out.append(f"- **Cliffhanger:** {p.cliffhanger}")
            out.append(f"- **Why it works:** {p.why}")
            out.append(f"- **Opens on:** {p.opening_line}")
            out.append(f"- **Cuts on:** {p.closing_line}")
            if j:
                out.append(f"- **Judge:** hook {j.get('hook')}, cliffhanger {j.get('cliffhanger')}, standalone "
                           f"{j.get('standalone')}, spoiler {j.get('spoiler')}. {j.get('critique', '')}")
            else:
                out.append("- **Judge:** not judged")
            if j.get("retention"):
                out.append(f"- **Predicted still watching (of 100, every 3 s):** {' → '.join(map(str, j['retention']))}"
                           f" · biggest drop: {j.get('biggest_drop', '')}")
            ad = self.ads.get(p.id)
            if ad:
                out.append(f"- **Overlay:** {ad.get('overlay')} · **B:** {ad.get('overlay_b')} · **Hinglish:** {ad.get('hinglish')}")
            if p.flags:
                out.append(f"- **Flags:** {'; '.join(p.flags)}")
            out.append("")
        out += ["## Rejected by rule checks", ""]
        if not self.rejected:
            out.append("- none")
        for r in self.rejected:
            out.append(f"- {r['id']}: {r['reason']}")
        return "\n".join(out)

    def weak_spots(self, ranked: list[Promo]) -> list[str]:
        judged = [p for p in ranked if p.judge]
        if not judged:
            return ["- Nothing was judged in this run."]
        n = len(judged)

        def count(f):
            return sum(1 for p in judged if f(p))
        out = [
            f"- **Weak hooks:** {count(lambda p: p.judge['hook'] <= 2)}/{n} promos the finder picked for their hook "
            "scored 2 or less with the judge, who sees only the clip.",
            f"- **Weak endings:** {count(lambda p: p.judge['cliffhanger'] <= WEAK_CLIFF)}/{n} end on a beat that feels "
            "resolved (cliffhanger 2 or less).",
            f"- **Needs context:** {count(lambda p: p.judge['standalone'] <= 3)}/{n} need earlier episodes to make sense "
            "(standalone 3 or less).",
            f"- **Spoilers:** {count(lambda p: p.judge['spoiler'])}/{n} give away a twist.",
            f"- **Silent openings:** {count(lambda p: not p.hook_in_3s)}/{n} have nothing said or done in the first 3 s, "
            "so the hook depends on a visual the script only describes.",
        ]
        rs = self.recut_stats
        if rs["weak_endings"]:
            out.append(f"- **Overshooting the cliffhanger:** the finder's first cut ended on a resolved beat in "
                       f"{rs['weak_endings']} clips. The re-cut step tried an earlier end line for {rs['recut_tried']} "
                       f"and kept {rs['recut_kept']} that the judge scored higher (see recuts.json).")
        trimmed = count(lambda p: any(f.startswith("trimmed") for f in p.flags))
        if trimmed:
            out.append(f"- **Trimmed:** {trimmed} promos ran past 60 s and were cut back, which can lose the cliffhanger.")
        reasons: dict[str, int] = {}
        for r in self.rejected:
            k = r["reason"].split(" (")[0]
            reasons[k] = reasons.get(k, 0) + 1
        if reasons:
            out.append("- **Thrown out by code checks:** " + ", ".join(
                f"{k} ({v})" for k, v in sorted(reasons.items(), key=lambda x: -x[1])))
        out.append("- **Lowest-scored picks** (what bad looks like):")
        for p in sorted(judged, key=lambda p: p.score)[:3]:
            out.append(f"  - {p.id} \"{p.title}\" (score {p.score}): {p.judge.get('critique', '')}")
        return out

    def experimental(self, ranked: list[Promo]) -> list[str]:
        """Report section for extras.py: predictions, clearly labelled as such."""
        if not self.extras:
            return []
        out = ["", "## Experimental: flash-forwards, predicted drop-off, ad packages", "",
               "These are AI predictions, not measurements. Use them to choose what to A/B test first."]
        fs = self.flash_stats
        if fs["tried"]:
            out.append(f"- **Flash-forward hooks:** tried a 1-4 s teaser of the clip's most shocking moment before "
                       f"{fs['tried']} top promos; {fs['kept']} scored higher than the plain version (IDs ending in `f`).")
        curves = [p for p in ranked if p.judge.get("retention")]
        if curves:
            avg = sum(p.judge["watch_through"] for p in curves) / len(curves)
            best = max(curves, key=lambda p: p.judge["watch_through"])
            worst = min(curves, key=lambda p: p.judge["watch_through"])
            out.append(f"- **Predicted drop-off** for the top {len(curves)}: on average {avg:.0f} of 100 scrollers "
                       f"watch to the end. Best hold: {best.id} ({best.judge['watch_through']}). Weakest: {worst.id} "
                       f"({worst.judge['watch_through']}, biggest drop {worst.judge.get('biggest_drop', '')}).")
        if self.ads:
            out += ["", "| Promo | On-screen text (A) | (B) | Hinglish | Best audience |", "|---|---|---|---|---|"]
            for p in ranked:
                a = self.ads.get(p.id)
                if a:
                    out.append(f"| {p.id} {p.title} | {a.get('overlay')} | {a.get('overlay_b')} | {a.get('hinglish')} | "
                               f"{a.get('audience')} |")
            out.append("")
            out.append("Captions and hashtags are in `ads.json`; cut lists for editing software are in `edl/`.")
        return out
