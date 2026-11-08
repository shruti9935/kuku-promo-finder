"""Offline tests: no API key or network needed.  Run:  python -m pytest -q"""
import json
import types
from pathlib import Path

import pytest

from promo_finder import parse as PA
from promo_finder import pipeline as PL
from promo_finder.llm import LLM, LLMConfig, extract_json

SAMPLE = """EPISODE NUMBER: Episode 7
EPISODE LOGLINE: A test episode.
CONTINUITY NOTES: none

--- SCRIPT START ---

SCENE 01
Scene heading: INT. HALL – NIGHT
Scene start: 00:00.00
Scene end: 01:10.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:03.00)
Wide shot of the hall.

**ACTION:**
Start: (00:03.00) | End: (00:05.00)
She slaps him.

**SOUND FX:**
Start: (00:03.20) | End: (00:03.60)
Crack of a slap.

**VILLAIN (O.S.) (Cold, Cruel):**
Start: (00:05.50) | End: (00:09.00)
"You will regret that."

**HEROINE (Defiant):**
Start: (00:09.50) | End: (00:14.00)
"I already regret marrying you."

**CAMERA:**
Start: (00:14.00) | End: (00:30.00)
Slow push in.

**VILLAIN (Whispering):**
Start: (00:30.00) | End: (00:36.00)
"Then you won't mind what happens next."

**CAMERA:**
Start: (00:36.00) | End: (00:37.00)
Close-up.

**HEROINE:**
Start: (00:40.00) | End: (00:45.00)
"Who are you really?"

**CAMERA:**
Start: (00:45.00) | End: (01:10.00)
Long hold on the door.

--- SCRIPT END ---
"""


@pytest.fixture
def ep():
    return PA.parse_text(SAMPLE, "episode-007-x.txt")


@pytest.fixture
def eps(ep):
    return {ep.number: ep}


# ---------- parse ----------

def test_parse_basic(ep):
    assert ep.number == 7 and ep.logline == "A test episode."
    assert [l.kind for l in ep.lines] == ["CAMERA", "ACTION", "SFX", "DIALOGUE", "DIALOGUE", "CAMERA",
                                          "DIALOGUE", "CAMERA", "DIALOGUE", "CAMERA"]
    v = ep.lines[3]
    assert v.speaker == "VILLAIN" and v.tone == "O.S.; Cold, Cruel"
    assert v.start == 5.5 and v.end == 9.0 and v.text == '"You will regret that."'
    assert ep.duration == 70.0


def test_parse_crlf_and_hours_and_filename_fallback():
    raw = SAMPLE.replace("EPISODE NUMBER: Episode 7\n", "").replace("\n", "\r\n").replace("(00:45.00)", "(0:00:45.00)")
    e = PA.parse_text(raw, "episode-012-whatever.txt")
    assert e.number == 12 and len(e.lines) == 10
    assert e.lines[-2].end == 45.0


def test_parse_garbage_gives_no_lines():
    assert PA.parse_text("hello\nworld", "x.txt").lines == []


def test_load_episodes_skips_empty(tmp_path):
    (tmp_path / "episode-002.txt").write_text(SAMPLE.replace("Episode 7", "Episode 2"), encoding="utf-8")
    (tmp_path / "episode-001.txt").write_text(SAMPLE.replace("Episode 7", "Episode 1"), encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a script", encoding="utf-8")
    assert [e.number for e in PA.load_episodes(tmp_path)] == [1, 2]


def test_timestamps_roundtrip():
    assert PA.to_seconds("01:12.30") == 72.3
    assert PA.fmt_ts(72.3) == "01:12.30"
    assert PA.to_seconds("1:00:00.00") == 3600.0


def test_render_has_line_ids(ep):
    txt = PA.render_for_llm(ep)
    assert txt.splitlines()[3].startswith("E07-L004 [00:05.50-00:09.00] VILLAIN (O.S.; Cold, Cruel):")
    assert PA.render_for_llm(ep, prefix=False).splitlines()[3].startswith("L004 ")
    assert ep.line("E07-L004").speaker == "VILLAIN" and ep.line("L4").speaker == "VILLAIN"


# ---------- resolve cut points ----------

def test_resolve_ids_to_times(ep):
    seg, err = PL.resolve_segment(ep, "L002", "L007")
    assert not err and seg.start == 3.0 and seg.end == 36.0


def test_resolve_never_cuts_mid_line(ep):
    # L005 is "I already regret..." 9.5-14.0; ending on L004 must not cut it if L005 started inside
    seg, _ = PL.resolve_segment(ep, "L001", "L003")   # camera 0-3 + action 3-5 + sfx
    assert seg.end == 5.0
    # start inside a dialogue line snaps back to the line's start
    ep.lines[5].start = 12.0                          # camera L006 now starts inside HEROINE 9.5-14.0
    seg, _ = PL.resolve_segment(ep, "L006", "L007")
    assert seg.start == 9.5


def test_resolve_finishes_short_camera_shot(ep):
    seg, _ = PL.resolve_segment(ep, "L004", "L007")  # ends 36.0, camera L008 36-37 starts exactly at end -> no change
    assert seg.end == 36.0
    ep.lines[7].start = 35.5                          # now the shot spans the cut and ends 1 s later
    seg, _ = PL.resolve_segment(ep, "L004", "L007")
    assert seg.end == 37.0


def test_resolve_bad_and_reversed_ids(ep):
    seg, err = PL.resolve_segment(ep, "L999", "L001")
    assert seg is None and "unknown line id" in err
    seg, err = PL.resolve_segment(ep, "L007", "L002")
    assert seg.start == 3.0 and seg.end == 36.0


# ---------- rules ----------

def mk(segs, kind="single"):
    return PL.Promo(id="T", kind=kind, title="t", segments=segs)


def test_rules_length(ep, eps):
    assert "too short" in PL.check_rules(mk([PL.Segment(7, 0, 10, "L001", "L002")]), eps)
    long = mk([PL.Segment(7, 0, 70, "L001", "L010")])
    assert PL.check_rules(long, eps) == ""                  # trimmed instead of thrown away
    assert long.duration <= 60 and long.segments[0].end == 45.0 and long.flags[0].startswith("trimmed from 70s")
    assert PL.check_rules(mk([PL.Segment(7, 3, 36, "L002", "L007")]), eps) == ""


def test_rules_order_and_overlap(eps):
    a, b = PL.Segment(7, 30, 45, "", ""), PL.Segment(7, 3, 14, "", "")
    assert "order" in PL.check_rules(mk([a, b]), eps)
    assert "overlap" in PL.check_rules(mk([PL.Segment(7, 3, 20, "", ""), PL.Segment(7, 15, 40, "", "")]), eps)


def test_hook_in_first_3s(eps):
    good = mk([PL.Segment(7, 3, 36, "L002", "L007")])
    PL.check_rules(good, eps)
    assert good.hook_in_3s and good.opening_line.startswith("action: She slaps")
    slow = mk([PL.Segment(7, 14, 45, "L006", "L009")])   # opens on a 16 s camera push
    PL.check_rules(slow, eps)
    assert not slow.hook_in_3s and "nothing said or done in first 3s" in slow.flags
    assert slow.closing_line == 'HEROINE: "Who are you really?"'


def test_clip_script_is_relative(eps):
    p = mk([PL.Segment(7, 3, 14, "", ""), PL.Segment(7, 30, 45, "", "")], "multi-segment")
    txt = PL.clip_script(p, eps)
    assert txt.splitlines()[0].startswith("  0.0s action: She slaps")
    assert "--- CUT ---" in txt and txt.strip().endswith("26.0s [CLIP ENDS]")


# ---------- score + dedupe ----------

def test_score_weights():
    p = mk([PL.Segment(30, 0, 30, "", "")])
    p.hook_in_3s = True
    p.judge = {"hook": 5, "cliffhanger": 5, "standalone": 5, "spoiler": False}
    assert PL.score(p) == 5.0
    p.judge["spoiler"] = True
    p.hook_in_3s = False
    assert PL.score(p) == 5.0 - PL.PENALTY_SPOILER - PL.PENALTY_NO_HOOK_IN_3S
    p.segments = [PL.Segment(3, 0, 30, "", "")]
    assert PL.score(p) == round(5.0 - PL.PENALTY_SPOILER - 0.5 + PL.EARLY_EPISODE_BONUS, 2)
    p.judge = {"hook": 5, "cliffhanger": 2, "standalone": 5, "spoiler": False}   # weak ending
    p.hook_in_3s = True
    assert PL.score(p) == round(0.4 * 5 + 0.3 * 2 + 0.3 * 5 - PL.PENALTY_WEAK_CLIFF + PL.EARLY_EPISODE_BONUS, 2)
    p.judge = {}
    assert PL.score(p) == 0.0


def test_dedupe_keeps_best_and_spares_montages(tmp_path):
    run = types.SimpleNamespace()
    a = mk([PL.Segment(5, 0, 30, "", "")]); a.judge = {"hook": 5, "cliffhanger": 5, "standalone": 5}; a.id = "A"
    b = mk([PL.Segment(5, 5, 35, "", "")]); b.judge = {"hook": 3, "cliffhanger": 3, "standalone": 3}; b.id = "B"
    c = mk([PL.Segment(5, 0, 12, "", ""), PL.Segment(9, 0, 12, "", "")], "montage"); c.judge = a.judge; c.id = "C"
    for p in (a, b, c):
        p.hook_in_3s = True
    kept = PL.Run.rank(run, [b, a, c])
    assert sorted(p.id for p in kept) == ["A", "C"] and b.duplicate_of == "A"


# ---------- LLM helpers ----------

def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": [1,2]} hope that helps') == {"a": [1, 2]}
    assert extract_json("no json here") is None
    assert extract_json("") is None


class FakeErr(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


def fake_client(script):
    """script: list of (model -> response or exception) consumed in order."""
    calls = []

    def create(**kw):
        calls.append(kw["model"])
        r = script.pop(0)
        if isinstance(r, Exception):
            raise r
        msg = types.SimpleNamespace(content=r)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))), calls


def make_llm(tmp_path, script, fallbacks="backup-model"):
    cfg = LLMConfig(provider="custom", model="main-model", base_url="http://x", api_key="k",
                    min_interval=0, cache_dir=str(tmp_path / "c"), fallbacks=fallbacks)
    llm = LLM(cfg)
    llm.client, calls = fake_client(script)
    return llm, calls


def test_llm_falls_back_when_overloaded(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    llm, calls = make_llm(tmp_path, [FakeErr(503)] * 4 + ['{"ok": true}'])
    data, _ = llm.chat_json("s", "u")
    assert data == {"ok": True} and calls[-1] == "backup-model" and llm.model_used == {"backup-model": 1}


def test_llm_retries_bad_json_then_caches(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    llm, calls = make_llm(tmp_path, ["not json", '{"x": 2}'])
    assert llm.chat_json("s", "u")[0] == {"x": 2}
    llm2, calls2 = make_llm(tmp_path, [])            # same cache dir, no responses left
    assert llm2.chat_json("s", "u")[0] == {"x": 2} and llm2.cache_hits == 1 and calls2 == []


def test_llm_auth_error_is_not_retried(tmp_path):
    llm, calls = make_llm(tmp_path, [FakeErr(401)])
    with pytest.raises(FakeErr):
        llm.chat_json("s", "u")
    assert calls == ["main-model"]


# ---------- whole pipeline with a misbehaving fake model ----------

class Liar(LLM):
    """Returns a mix of good, invented and broken answers."""
    def __init__(self):
        self.cfg = LLMConfig(provider="fake", model="fake")
        self.calls = self.cache_hits = 0
        self.usage, self.model_used = {}, {}

    def chat_json(self, system, user, use_cache=True):
        self.calls += 1
        if "HEAD-TO-HEAD" in user:                                                  # reverse the score order
            import re
            d = {"order": list(reversed(re.findall(r"=== CLIP (\S+)", user))), "best_reason": "r"}
        elif "AVAILABLE MOMENTS" in user:
            d = {"montages": [{"title": "m", "moment_ids": ["M002", "M001"]},       # wrong order
                              {"title": "ghost", "moment_ids": ["M999"]}]}          # invented
        elif "CLIPS:" in user:
            import re
            ids = re.findall(r"=== CLIP (\S+)", user)
            d = {"ratings": [{"id": i, "hook": "4", "cliffhanger": 9, "standalone": 0, "spoiler": "yes"}
                             for i in ids[:-1]]}                                    # skips the last clip
        else:
            import re
            promos, moments = [], []
            for n in re.findall(r"=== EPISODE (\d+) ===", user):
                e = f"E{int(n):02d}-"
                promos += [
                    {"title": "ok", "segments": [{"start_line": e + "L002", "end_line": e + "L007"}]},
                    {"title": "invented", "segments": [{"start_line": e + "L500", "end_line": e + "L510"}]},
                    {"title": "short", "segments": [{"start_line": e + "L002", "end_line": e + "L003"}]},
                    {"title": "no segs", "episode": int(n)},
                    {"title": "ghost ep", "segments": [{"start_line": "E99-L002", "end_line": "E99-L007"}]},
                    {"title": "spans", "segments": [{"start_line": e + "L002", "end_line": "E99-L007"}]}]
                moments += [{"start_line": e + "L002", "end_line": e + "L005"},
                            {"start_line": e + "L007", "end_line": e + "L009"}]
            d = {"promos": promos, "key_moments": moments}
        return d, json.dumps(d)


def test_pipeline_survives_bad_model_output(tmp_path):
    d = tmp_path / "show"
    d.mkdir()
    for n in (1, 2):
        (d / f"episode-00{n}.txt").write_text(SAMPLE.replace("Episode 7", f"Episode {n}"), encoding="utf-8")
    out = PL.Run(str(d), Liar(), out_root=str(tmp_path / "out"), log=lambda m: None).execute()
    meta = json.loads((out / "run_meta.json").read_text())
    ranked = json.loads((out / "promos_ranked.json").read_text())
    rejected = json.loads((out / "rejected.json").read_text())
    reasons = " ".join(r["reason"] for r in rejected)
    assert "unknown line id" in reasons and "too short" in reasons and "no segments" in reasons
    assert "episode not in this batch" in reasons and "segment spans two episodes" in reasons
    assert "fewer than 2 valid moments" in reasons
    assert meta["passed_rules"] == 2 and meta["montages_passed"] == 1 and meta["batch_size"] == 5
    for p in ranked:                                # judge scores coerced + clamped
        if p["judge"]:
            assert p["judge"]["hook"] == 4 and p["judge"]["cliffhanger"] == 5 and p["judge"]["standalone"] == 1
    m = next(p for p in ranked if p["kind"] == "montage")
    assert "moments reordered into story order" in m["flags"]
    assert (out / "report.md").exists() and (out / "promos_ranked.csv").exists()


def test_empty_folder_errors(tmp_path):
    with pytest.raises(ValueError):
        PL.Run(str(tmp_path), Liar(), out_root=str(tmp_path / "o"))


def test_parse_range():
    from run import parse_range
    assert parse_range("1-3,7") == [1, 2, 3, 7] and parse_range(None) is None


def test_evaluate(tmp_path, capsys):
    import evaluate
    run = tmp_path / "r"
    run.mkdir()
    (run / "promos_ranked.json").write_text(json.dumps([
        {"id": "E01-P1", "rank": 1, "episodes": [1], "segments": [{"episode": 1, "start_sec": 10, "end_sec": 40}]},
        {"id": "E01-P2", "rank": 2, "episodes": [1], "segments": [{"episode": 1, "start_sec": 80, "end_sec": 110}]}]))
    gold = tmp_path / "g.json"
    gold.write_text(json.dumps([{"episode": 1, "start": "00:12.00", "end": "00:42.00"},
                                {"episode": 1, "start": 200, "end": 230}]))
    evaluate.main(str(run), str(gold))
    out = capsys.readouterr().out
    assert "Recall: 1/2" in out and "Precision: 1/2" in out


class QuotaAfter(Liar):
    """Works for n calls, then reports the daily quota is gone."""
    def __init__(self, n):
        super().__init__()
        self.left = n

    def chat_json(self, system, user, use_cache=True):
        from promo_finder.llm import QuotaExhausted
        if self.left <= 0:
            raise QuotaExhausted("daily quota used up")
        self.left -= 1
        return super().chat_json(system, user, use_cache)


def test_quota_stops_run_but_keeps_finished_work(tmp_path):
    d = tmp_path / "show"
    d.mkdir()
    for n in (1, 2, 3):
        (d / f"episode-00{n}.txt").write_text(SAMPLE.replace("Episode 7", f"Episode {n}"), encoding="utf-8")
    llm = QuotaAfter(1)                      # only the first find batch succeeds
    run = PL.Run(str(d), llm, out_root=str(tmp_path / "out"), batch_size=1, log=lambda m: None)
    out = run.execute()
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["stopped_early"] and llm.calls == 1          # no calls wasted after the quota error
    assert meta["passed_rules"] == 1                         # episode 1's promo is kept
    ranked = json.loads((out / "promos_ranked.json").read_text())
    assert ranked and "not judged: quota used up" in ranked[0]["flags"]


def test_batching_groups_episodes(tmp_path):
    d = tmp_path / "show"
    d.mkdir()
    for n in range(1, 6):
        (d / f"episode-00{n}.txt").write_text(SAMPLE.replace("Episode 7", f"Episode {n}"), encoding="utf-8")
    llm = Liar()
    out = PL.Run(str(d), llm, out_root=str(tmp_path / "out"), batch_size=2, judge_batch=50,
                 log=lambda m: None).execute()
    raws = sorted(p.name for p in (out / "raw").iterdir())
    assert [r for r in raws if r.startswith("find")] == ["find_ep001-002.json", "find_ep003-004.json", "find_ep005-005.json"]
    assert len([r for r in raws if r.startswith("judge_singles")]) == 1
    ids = [p["id"] for p in json.loads((out / "promos_ranked.json").read_text()) if p["kind"] != "montage"]
    assert sorted(ids) == ["E01-P1", "E02-P1", "E03-P1", "E04-P1", "E05-P1"]


# ---------- v3 rules: cuts, trimming, shortlist, separate judge ----------

def test_too_many_cuts_and_jump_cuts(eps):
    four = [PL.Segment(7, 0, 5, "", ""), PL.Segment(7, 5.5, 14, "", ""), PL.Segment(7, 30, 36, "", ""),
            PL.Segment(7, 40, 45, "", "")]
    assert "too many cuts" in PL.check_rules(mk(four), eps)
    jump = [PL.Segment(7, 3, 5, "", ""), PL.Segment(7, 9.5, 45, "", "")]
    assert "jump cut" in PL.check_rules(mk(jump, "multi-segment"), eps)


def test_trim_that_cannot_fit_is_rejected(eps):
    # one 70 s camera-only stretch: no spoken line ends inside 60 s after the start
    p = mk([PL.Segment(7, 45, 115, "L010", "L010")])
    assert "too long" in PL.check_rules(p, eps)


def _judged(pid, ep, hook, spoiler=False):
    p = mk([PL.Segment(ep, 0, 30, "", "")])
    p.id, p.hook_in_3s = pid, True
    p.judge = {"hook": hook, "cliffhanger": 4, "standalone": 4, "spoiler": spoiler, "critique": "c"}
    p.score = PL.score(p)
    return p


def test_shortlist_one_per_episode_and_no_spoilers():
    ranked = [_judged("A", 3, 5, spoiler=True), _judged("B", 3, 5), _judged("C", 3, 4), _judged("D", 8, 4)]
    assert [p.id for p in PL.Run.shortlist(ranked)] == ["B", "D"]


class Grader(Liar):
    def chat_json(self, system, user, use_cache=True):
        assert "CLIPS:" in user, "finder work was sent to the judge"
        return super().chat_json(system, user, use_cache)


def test_separate_judge_model_is_used_and_reported(tmp_path):
    d = tmp_path / "show"
    d.mkdir()
    for n in (1, 2):   # the fake judge skips the last clip of each batch, so give it two
        (d / f"episode-00{n}.txt").write_text(SAMPLE.replace("Episode 7", f"Episode {n}"), encoding="utf-8")
    finder, judge = Liar(), Grader()
    judge.cfg = LLMConfig(provider="fake", model="judge-model")
    out = PL.Run(str(d), finder, out_root=str(tmp_path / "out"), log=lambda m: None, judge_llm=judge).execute()
    meta = json.loads((out / "run_meta.json").read_text())
    assert judge.calls >= 1 and meta["judge_llm"]["model"] == "judge-model" and meta["llm_calls"] == finder.calls + judge.calls
    rep = (out / "report.md").read_text(encoding="utf-8")
    assert "## Where the tool picks badly" in rep and "Weak hooks" in rep and "judge: `judge-model`" in rep


def test_judge_config_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "finder-m")
    monkeypatch.setenv("JUDGE_MODEL", "judge-m")
    f, j = LLMConfig.from_env(), LLMConfig.from_env(prefix="JUDGE_")
    assert f.model == "finder-m" and j.model == "judge-m" and j.provider == "gemini" and j.api_key == "k"


def test_head_to_head_reorders_top_only():
    run = types.SimpleNamespace(aborted="", errors=[], log=lambda m: None, eps={7: PA.parse_text(SAMPLE, "e7.txt")},
                                judge_llm=Liar(), _raw=lambda *a: None, _stop=lambda *a: None)
    ranked = [_judged(c, 7, 5) for c in "ABCD"]
    for p in ranked:
        p.segments = [PL.Segment(7, 3, 36, "L002", "L007")]
    out = PL.Run.head_to_head(run, ranked)
    assert [p.id for p in out] == ["D", "C", "B", "A"] and out[0].judge["head_to_head"] == 1


def test_clean_id():
    for raw in ("E03-P1", "CLIP E03-P1", "[E03-P1]", " clip: E03-P1 ", "X-2"):
        assert PL.clean_id(raw) in ("E03-P1", "X-2")


class Recutter(Liar):
    """Judge says every original ends flat (cliffhanger 1) and every re-cut ends hot (5)."""
    def chat_json(self, system, user, use_cache=True):
        import re
        self.calls += 1
        if "RE-CUT" in user:
            ids = re.findall(r"=== CLIP (\S+)", user)
            d = {"recuts": [{"id": f"CLIP {i}", "end_line": f"E{int(i[1:3]):02d}-L006", "why": "threat"} for i in ids]}
        elif "CLIPS:" in user and "HEAD-TO-HEAD" not in user:
            ids = re.findall(r"=== CLIP (\S+)", user)
            d = {"ratings": [{"id": i, "hook": 4, "cliffhanger": 5 if i.endswith("r") else 1, "standalone": 4,
                              "spoiler": False, "critique": "c"} for i in ids]}
        else:
            self.calls -= 1
            return super().chat_json(system, user, use_cache)
        return d, json.dumps(d)


def test_recut_replaces_flat_ending_with_earlier_cut(tmp_path):
    d = tmp_path / "show"
    d.mkdir()
    (d / "episode-001.txt").write_text(SAMPLE.replace("Episode 7", "Episode 1"), encoding="utf-8")
    out = PL.Run(str(d), Recutter(), out_root=str(tmp_path / "out"), log=lambda m: None, max_montages=0).execute()
    ranked = json.loads((out / "promos_ranked.json").read_text())
    meta = json.loads((out / "run_meta.json").read_text())
    r = next(p for p in ranked if p["id"] == "E01-P1r")          # L002..L007 (3-36 s) re-cut to end on L006 (30 s)
    assert r["segments"][0]["end_sec"] < 36 and r["judge"]["cliffhanger"] == 5
    assert meta["recut"]["recut_kept"] >= 1 and json.loads((out / "recuts.json").read_text())
    assert "Overshooting the cliffhanger" in (out / "report.md").read_text(encoding="utf-8")


# ---------- experimental extras ----------

from promo_finder import extras as EX   # noqa: E402


def test_flash_forward_rules(eps):
    teaser, body = PL.Segment(7, 5.5, 9, "L004", "L004"), PL.Segment(7, 3, 36, "L002", "L007")
    ff = mk([teaser, body], "flash-forward")
    assert PL.check_rules(ff, eps) == "" and ff.hook_in_3s and any("flash-forward" in f for f in ff.flags)
    long_teaser = mk([PL.Segment(7, 14, 30, "L006", "L006"), body], "flash-forward")
    assert "teaser longer" in PL.check_rules(long_teaser, eps)
    assert "order" in PL.check_rules(mk([teaser, body]), eps)          # same cut as a normal promo: out of order


def test_clean_curve():
    assert EX.clean_curve([90, "70", 75, None, 20], 13) == [90, 70, 70, 20, 20]   # never rises, padded to 5 windows
    assert EX.clean_curve([120, -5], 6) == [100, 0]
    assert EX.clean_curve("junk", 30) == []


def test_edl_timecodes():
    p = mk([PL.Segment(12, 43.7, 50.0, "L010", "L014"), PL.Segment(12, 60.0, 70.0, "L020", "L025")], "multi-segment")
    p.id, p.title = "E12-P1", "Ultimatum"
    text = EX.edl(p)
    assert "001  EP012    V     C        00:00:43:21 00:00:50:00 00:00:00:00 00:00:06:09" in text
    assert "002  EP012    V     C        00:01:00:00 00:01:10:00 00:00:06:09 00:00:16:09" in text


class Showman(Liar):
    """Fake AI for the extras: picks a teaser, draws curves, writes ad copy; the judge loves teasers."""
    def chat_json(self, system, user, use_cache=True):
        import re
        ids = re.findall(r"=== (?:CLIP|PROMO) (\S+?)[:\s]", user)
        if "FLASH-FORWARD" in user:
            d = {"teasers": [{"id": i, "line": f"E{int(i[1:3]):02d}-L004", "why": "the threat"} for i in ids]}
        elif "PREDICTED DROP-OFF" in user:
            d = {"curves": [{"id": i, "still_watching": [80, 60, 50, 45, 40, 38, 35, 30, 30, 28, 25, 25], "biggest_drop": "0-3 s"} for i in ids]}
        elif "AD PACKAGE" in user:
            d = {"ads": [{"id": i, "overlay": "She slapped him 😱", "overlay_b": "Would you forgive him?", "caption": "Watch Episode 1 now",
                          "hashtags": ["#drama"], "hinglish": "Usne thappad maara 😱", "audience": "romance fans"} for i in ids]}
        elif "CLIPS:" in user and "HEAD-TO-HEAD" not in user:
            d = {"ratings": [{"id": i, "hook": 5 if i.endswith("f") else 3, "cliffhanger": 4, "standalone": 4, "spoiler": False,
                              "critique": "c"} for i in ids]}
        else:
            return super().chat_json(system, user, use_cache)
        self.calls += 1
        return d, json.dumps(d)


def test_extras_end_to_end(tmp_path):
    d = tmp_path / "show"
    d.mkdir()
    (d / "episode-001.txt").write_text(SAMPLE.replace("Episode 7", "Episode 1"), encoding="utf-8")
    out = PL.Run(str(d), Showman(), out_root=str(tmp_path / "out"), log=lambda m: None, max_montages=0).execute()
    ranked = json.loads((out / "promos_ranked.json").read_text(encoding="utf-8"))
    meta = json.loads((out / "run_meta.json").read_text())
    ff = next(p for p in ranked if p["kind"] == "flash-forward")
    assert ff["id"] == "E01-P1f" and ff["segments"][0]["start_sec"] == 5.5 and meta["flash_forward"]["kept"] == 1
    assert ff["judge"]["retention"][0] == 80 and ff["judge"]["watch_through"] == 25
    assert ff["ad"]["hinglish"].startswith("Usne") and (out / "edl" / "E01-P1f.edl").exists()
    rep = (out / "report.md").read_text(encoding="utf-8")
    assert "## Experimental" in rep and "Flash-forward hooks" in rep and "Predicted drop-off" in rep
    assert "predicted_watch_through_pct" in (out / "promos_ranked.csv").read_text(encoding="utf-8")


def test_extras_can_be_switched_off(tmp_path):
    d = tmp_path / "show"
    d.mkdir()
    (d / "episode-001.txt").write_text(SAMPLE.replace("Episode 7", "Episode 1"), encoding="utf-8")
    out = PL.Run(str(d), Showman(), out_root=str(tmp_path / "out"), log=lambda m: None, extras=False).execute()
    assert not (out / "edl").exists() and "## Experimental" not in (out / "report.md").read_text(encoding="utf-8")
