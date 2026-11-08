"""Promo Finder: Streamlit app.  Run:  streamlit run app.py"""
import hashlib
import json
import os
from pathlib import Path

import streamlit as st

from promo_finder.llm import LLM, LLMConfig, PRESETS
from promo_finder.parse import load_episodes
from promo_finder.pipeline import Promo, Segment, Run, clip_script
from run import load_dotenv, parse_range

load_dotenv()
try:  # hosted app (Streamlit Cloud): keys come from the app's secrets
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:
    pass
st.set_page_config(page_title="Promo Finder", layout="wide")
OUT = Path("outputs")


def promo_from_dict(d: dict) -> Promo:
    segs = [Segment(s["episode"], s["start_sec"], s["end_sec"], s["start_line"], s["end_line"]) for s in d["segments"]]
    p = Promo(id=d["id"], kind=d["kind"], title=d["title"], segments=segs)
    return p


def show_run(run_dir: Path):
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    promos = json.loads((run_dir / "promos_ranked.json").read_text(encoding="utf-8"))
    eps = {}
    if meta.get("input_dir") and Path(meta["input_dir"]).exists():
        eps = {e.number: e for e in load_episodes(meta["input_dir"])}

    c = st.columns(5)
    c[0].metric("Promos", meta["final_promos"])
    c[1].metric("Candidates", meta["candidates_found"])
    c[2].metric("Failed rules", sum(meta["rejected_by_reason"].values()))
    c[3].metric("Duplicates removed", meta["duplicates_removed"])
    c[4].metric("Finder / judge", f'{meta["llm"]["model"]} / {(meta.get("judge_llm") or meta["llm"])["model"]}')
    if meta.get("stopped_early"):
        st.error("Run stopped early: " + meta["stopped_early"][:300])
    elif meta.get("errors"):
        st.warning("Some steps failed: " + "; ".join(meta["errors"][:5]))
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    top = report.split("## All promos")[0].split("## Post these first", 1)[-1]
    with st.expander("Summary: post these first · where the tool picks badly", expanded=True):
        st.markdown("#### Post these first" + top)

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download CSV", (run_dir / "promos_ranked.csv").read_bytes(), "promos_ranked.csv")
    d2.download_button("Download JSON", (run_dir / "promos_ranked.json").read_bytes(), "promos_ranked.json")
    d3.download_button("Download report", report.encode("utf-8"), "report.md")
    all_kinds = ["single", "flash-forward", "multi-segment", "montage"]
    kinds = st.multiselect("Show", all_kinds, all_kinds)
    for p in promos:
        if p["kind"] not in kinds:
            continue
        j = p.get("judge") or {}
        segs = " + ".join(f"Ep {s['episode']} {s['start']}→{s['end']}" for s in p["segments"])
        with st.container(border=True):
            st.markdown(f"**#{p['rank']} · {p['title']}**  ·  score **{p['score']}**  ·  {p['kind']}  ·  {p['duration']:.0f}s")
            st.markdown(f"`{segs}`")
            a, b = st.columns(2)
            a.markdown(f"**Hook:** {p['hook']}  \n**Cliffhanger:** {p['cliffhanger']}  \n**Why:** {p['why']}")
            b.markdown(f"**Judge:** hook {j.get('hook')}/5 · cliff {j.get('cliffhanger')}/5 · standalone "
                       f"{j.get('standalone')}/5{' · ⚠ spoiler' if j.get('spoiler') else ''}  \n{j.get('critique', '')}")
            if p.get("flags"):
                b.caption("Flags: " + "; ".join(p["flags"]))
            ad = p.get("ad") or {}
            if ad:
                rows = [f"**On-screen text:** {ad.get('overlay')}", f"**B version:** {ad.get('overlay_b')}",
                        f"**Hinglish:** {ad.get('hinglish')}",
                        f"**Caption:** {ad.get('caption')} {' '.join(ad.get('hashtags') or [])}",
                        f"**Best audience:** {ad.get('audience')}"]
                a.markdown("  \n".join(rows))
            curve = j.get("retention")
            if curve:
                b.caption(f"Predicted still watching, of 100 scrollers (AI prediction) · ends at {curve[-1]} · "
                          f"biggest drop: {j.get('biggest_drop', '')}")
                b.line_chart({"still watching": [100] + curve}, height=140)
            edl_file = run_dir / "edl" / f"{p['id']}.edl"
            if edl_file.exists():
                b.download_button("Cut list for Premiere / Resolve (.edl)", edl_file.read_bytes(), edl_file.name,
                                  key=f"edl-{run_dir.name}-{p['id']}")
            if eps:
                with st.expander("Clip script"):
                    st.code(clip_script(promo_from_dict(p), eps), language=None)


st.title("Promo Finder")
st.caption("Finds 20-60 s promo clips in microdrama screenplays: hook in the first 3 s, cliffhanger at the cut, "
           "makes sense to a stranger.")

with st.sidebar:
    st.header("Model")
    provider = st.selectbox("Provider", list(PRESETS), index=list(PRESETS).index(os.getenv("LLM_PROVIDER", "gemini")))
    model = st.text_input("Model", value=os.getenv("LLM_MODEL", "") or PRESETS[provider][2])
    base_url = st.text_input("Base URL (optional)", value="")
    key = st.text_input("API key (blank = from .env)", type="password")
    judge_model = st.text_input("Judge model (blank = same provider, different model if set in .env)",
                                value=os.getenv("JUDGE_MODEL", ""))
    st.header("Run")
    ep_range = st.text_input("Episodes (blank = all)", placeholder="1-5")
    max_promos = st.slider("Candidates per episode", 1, 5, 3)
    max_montages = st.slider("Cross-episode montages", 0, 8, 5)

tab_new, tab_old = st.tabs(["New run", "Past runs"])

with tab_new:
    src = st.radio("Screenplays", ["Upload .txt files", "Folder on this machine"], horizontal=True)
    input_dir = None
    if src == "Upload .txt files":
        files = st.file_uploader("Episode screenplays", type="txt", accept_multiple_files=True)
        if files:
            h = hashlib.sha256(b"".join(sorted(f.name.encode() + f.getvalue() for f in files))).hexdigest()[:12]
            d = Path(".cache/uploads") / h
            d.mkdir(parents=True, exist_ok=True)
            for f in files:
                (d / Path(f.name).name).write_bytes(f.getvalue())
            input_dir = str(d)
    else:
        default = next((str(p) for p in Path("data").glob("*") if p.is_dir()), "data")
        input_dir = st.text_input("Folder", value=default)

    if input_dir and st.button("Find promos", type="primary"):
        cfg = LLMConfig.from_env(provider=provider, model=model, base_url=base_url, api_key=key)
        log = st.status("Running…", expanded=True)
        try:
            judge = None
            if judge_model and judge_model != cfg.model:
                judge = LLM(LLMConfig.from_env(prefix="JUDGE_", provider=provider, model=judge_model,
                                               base_url=base_url, api_key=key))
            run = Run(input_dir, LLM(cfg), out_root=str(OUT), episodes=parse_range(ep_range),
                      max_promos=max_promos, max_montages=max_montages, log=log.write, judge_llm=judge)
            out = run.execute()
            log.update(label="Done", state="complete", expanded=False)
            st.session_state["last_run"] = str(out)
        except Exception as e:
            log.update(label="Failed", state="error")
            st.error(str(e))
    if st.session_state.get("last_run"):
        show_run(Path(st.session_state["last_run"]))

with tab_old:
    runs = sorted([p for p in OUT.rglob("run-*") if (p / "run_meta.json").exists()],
                  key=lambda p: p.name, reverse=True)
    if not runs:
        st.info("No runs yet.")
    else:
        pick = st.selectbox("Run", runs, format_func=lambda p: str(p.relative_to(OUT)))
        show_run(pick)
