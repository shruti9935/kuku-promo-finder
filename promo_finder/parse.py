"""Read Kuku screenplay .txt files into timestamped lines. Plain code, no AI.

Each block in a script looks like:

    **MATRIARCH (Cold, Commanding):**
    Start: (00:11.30) | End: (00:13.10)
    "You should turn around, Juliana."
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

HEADER_RE = re.compile(r"^\*\*(?P<label>[^*]+?):\*\*\s*$")
TIME_RE = re.compile(r"Start:\s*\((?P<s>[\d:.]+)\)\s*\|\s*End:\s*\((?P<e>[\d:.]+)\)")
EP_NUM_RE = re.compile(r"EPISODE NUMBER:\s*Episode\s*(\d+)", re.I)
LOGLINE_RE = re.compile(r"EPISODE LOGLINE:\s*(.+)")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)")
HEADING_RE = re.compile(r"^Scene heading:\s*(.+)")

NON_SPEAKER = {"CAMERA", "ACTION", "SOUND FX", "SFX", "MUSIC", "TEXT ON SCREEN", "SUPER"}


def to_seconds(ts: str) -> float:
    """'01:12.30' -> 72.3"""
    parts = ts.split(":")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return round(secs, 2)


def fmt_ts(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Line:
    idx: int            # 1-based position inside the episode
    kind: str           # DIALOGUE | ACTION | CAMERA | SFX | OTHER
    speaker: str        # character name for DIALOGUE, else ""
    tone: str           # e.g. "Cold, Commanding"
    start: float
    end: float
    text: str
    scene: int = 0

    @property
    def lid(self) -> str:
        return f"L{self.idx:03d}"


@dataclass
class Episode:
    number: int
    logline: str
    file: str
    lines: list[Line] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((l.end for l in self.lines), default=0.0)

    def line(self, lid: str) -> Line | None:
        m = re.search(r"L\s*(\d+)", str(lid), re.I) or re.search(r"(\d+)", str(lid))
        if not m:
            return None
        i = int(m.group(1))
        return self.lines[i - 1] if 1 <= i <= len(self.lines) else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        return d


def _classify(label: str) -> tuple[str, str, str]:
    """'LADY JULIANA (Trembling, Pleading)' -> ('DIALOGUE', 'LADY JULIANA', 'Trembling, Pleading')"""
    # "THE BEARDED STRANGER (O.S.) (Stern, Accusing)" -> name + "O.S.; Stern, Accusing"
    name = label.split("(", 1)[0].strip().upper() or label.strip().upper()
    tone = "; ".join(p.strip() for p in re.findall(r"\(([^)]*)\)", label) if p.strip())
    if name == "CAMERA":
        return "CAMERA", "", tone
    if name == "ACTION":
        return "ACTION", "", tone
    if name in ("SOUND FX", "SFX"):
        return "SFX", "", tone
    if name in NON_SPEAKER:
        return "OTHER", name, tone
    return "DIALOGUE", name, tone


def parse_text(raw: str, file: str = "") -> Episode:
    lines = raw.splitlines()
    num_m = EP_NUM_RE.search(raw)
    log_m = LOGLINE_RE.search(raw)
    if num_m:
        number = int(num_m.group(1))
    else:  # fall back to "episode-007-..." in the file name
        fm = re.search(r"(\d+)", Path(file).name)
        number = int(fm.group(1)) if fm else 0
    ep = Episode(number=number, logline=(log_m.group(1).strip() if log_m else ""), file=Path(file).name)

    scene = 0
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        sm = SCENE_RE.match(s)
        if sm:
            scene = int(sm.group(1))
        hm = HEADER_RE.match(s)
        if hm:
            kind, speaker, tone = _classify(hm.group("label"))
            # next non-empty line should be the timing line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            tm = TIME_RE.search(lines[j]) if j < len(lines) else None
            if tm:
                # text = following lines until a blank line or next header
                k = j + 1
                text_parts = []
                while k < len(lines) and lines[k].strip() and not HEADER_RE.match(lines[k].strip()):
                    text_parts.append(lines[k].strip())
                    k += 1
                ep.lines.append(Line(
                    idx=len(ep.lines) + 1, kind=kind, speaker=speaker, tone=tone,
                    start=to_seconds(tm.group("s")), end=to_seconds(tm.group("e")),
                    text=" ".join(text_parts), scene=scene,
                ))
                i = k
                continue
        i += 1
    return ep


def load_episodes(folder: str | Path) -> list[Episode]:
    folder = Path(folder)
    eps = [parse_text(p.read_text(encoding="utf-8", errors="replace"), str(p))
           for p in sorted(folder.glob("*.txt"))]
    eps = [e for e in eps if e.lines]
    eps.sort(key=lambda e: e.number)
    return eps


def render_for_llm(ep: Episode, camera_chars: int = 140, prefix: bool = True) -> str:
    """Compact, line-numbered script the model reads. Line IDs let the model point at
    moments without inventing timestamps; code turns IDs back into times."""
    out = []
    for l in ep.lines:
        t = f"[{fmt_ts(l.start)}-{fmt_ts(l.end)}]"
        lid = f"E{ep.number:02d}-{l.lid}" if prefix else l.lid
        if l.kind == "DIALOGUE":
            tone = f" ({l.tone})" if l.tone else ""
            out.append(f"{lid} {t} {l.speaker}{tone}: {l.text}")
        elif l.kind == "CAMERA":
            txt = l.text if len(l.text) <= camera_chars else l.text[:camera_chars] + "…"
            out.append(f"{lid} {t} (camera) {txt}")
        else:
            label = l.speaker or l.kind
            out.append(f"{lid} {t} {label}: {l.text}")
    return "\n".join(out)
