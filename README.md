# Promo Finder

**Promo Finder reads the scripts of a short vertical drama (a "microdrama") and picks the best 20–60 second clips to use as ads on Instagram Reels, TikTok and YouTube Shorts.**

You give it a folder of episode scripts. It gives back a ranked list of promos. Each promo has:

- which episode(s) it comes from, and the exact **start and end time** of every piece to cut
- the **hook**: what happens in the first 3 seconds to stop a scrolling viewer
- the **cliffhanger**: the unresolved moment the clip cuts on
- **one line on why it works**, plus scores and warnings from an independent AI reviewer

Built for the Kuku AI Generalist assignment (Problem 1, "Find the Promos").

> **New here?** Open [`docs/how-it-works.html`](docs/how-it-works.html) in a browser for a visual, step-by-step walkthrough.
> The 2-page solution write-up is [`docs/Promo_Finder_Solution.pdf`](docs/Promo_Finder_Solution.pdf).

---

## Contents

1. [What makes a good promo](#1-what-makes-a-good-promo)
2. [Quick start](#2-quick-start)
3. [What the output looks like](#3-what-the-output-looks-like)
4. [How it works](#4-how-it-works)
   - [Experimental extras](#experimental-extras-flash-forwards-drop-off-curves-ad-packages)
5. [Why it is built this way](#5-why-it-is-built-this-way)
6. [What each file does](#6-what-each-file-does)
7. [Settings and command-line options](#7-settings-and-command-line-options)
8. [Output files](#8-output-files)
9. [Testing and checking quality](#9-testing-and-checking-quality)
10. [Results on the full show](#10-results-on-the-full-show)
11. [Known weak spots](#11-known-weak-spots)
12. [Confidential data](#12-confidential-data)

---

## 1. What makes a good promo

A promo is a short clip cut from the show and posted as an ad. Its only job is to make a stranger tap through to Episode 1. Following the brief, a good promo:

| Rule | What it means | How the tool checks it |
|---|---|---|
| **20–60 seconds** | Long enough to tell a mini-story, short enough for a feed | Code measures it exactly from the script timestamps |
| **Hook in the first 3 s** | Opens on a slap, a reveal, an insult, a threat, not on scenery | Code checks something is said or done in 0–3 s; the AI reviewer scores how strong it is |
| **Cliffhanger at the cut** | Ends on an open question, not on a resolved moment | The AI reviewer scores it; weak endings are automatically re-cut |
| **Makes sense to a stranger** | No need to have seen earlier episodes | The AI reviewer sees *only the clip*, nothing else about the show |
| **No big spoilers** | Doesn't give away the twist new viewers should discover | The AI reviewer flags it; spoilers go to a separate "hold for later" list |

A promo can be one continuous stretch, up to 3 stretches from the same episode, or a **montage** of 2–4 moments from different episodes.

---

## 2. Quick start

You need **Python 3.10+** and a free **Google Gemini API key** from <https://aistudio.google.com/apikey>.

```bash
# 1. install the two dependencies
pip install -r requirements.txt

# 2. add your key
cp .env.example .env            # Windows: copy .env.example .env
#    then open .env and paste your key after GEMINI_API_KEY=

# 3. try it on the small test show that ships with the repo (2 episodes, about 1 minute)
python run.py --input tests/fixtures/new_show

# 4. run it on a full show: a folder with one .txt script per episode
python run.py --input "data/<show folder>"

# 5. or use the web app: upload scripts and browse the ranked promos
streamlit run app.py
```

Each run creates a new folder `outputs/run-<date>-<time>/`. Start with **`report.md`** in that folder.

**Re-runs are free:** every AI reply is saved to `.cache/`, so running the same thing again costs no API calls. Use `--no-cache` to force fresh answers.

### Input format

One `.txt` file per episode, in Kuku's screenplay format: a header with the episode number and a one-line summary, then blocks like this:

```
EPISODE NUMBER: Episode 1
EPISODE LOGLINE: A humiliated waitress at a billionaire's engagement party is revealed to be his secret wife.

**VANESSA (Cruel):**
Start: (00:11.00) | End: (00:14.00)
"Oops. Now you finally match the help."

**CAMERA:**
Start: (00:14.00) | End: (00:17.00)
Guests laugh and raise phones to film.
```

Speaker blocks become dialogue. `CAMERA`, `ACTION` and `SOUND FX` blocks become visual and sound cues. See [`tests/fixtures/new_show/`](tests/fixtures/new_show/) for two complete example episodes.

---

## 3. What the output looks like

A real promo from the test show (`outputs/new-show-final/`):

```
#3  Who Did This To My Wife          single · 32 s · score 4.85
    Episode 1: 00:08.00 → 00:40.00   (lines L006–L020)
    Hook:        A guest pours wine over a waitress while the crowd laughs and films.
    Cliffhanger: The billionaire asks "Who did this to my wife?"; a glass shatters in silence.
    Why:         Instant visual humiliation followed by an immediate status power flip.
    Judge:       hook 4/5 · cliffhanger 5/5 · standalone 5/5 · no spoiler
                 "A strong visual hook and immediate conflict lead to a powerful, perfectly timed cliffhanger."
```

And the clip exactly as a viewer meets it (times from the first frame), which is also what the AI reviewer sees:

```
  0.0s (camera) Close-up on Vanessa tipping a full glass of red wine down Maya's white shirt.
  0.2s action: Vanessa pours the wine slowly over Maya, smiling.
  3.0s VANESSA: "Oops. Now you finally match the help."
  ...
 19.5s ETHAN: "Maya?"
 27.0s ETHAN: "Who did this to my wife?"
 30.6s sfx: The whole party falls silent. A glass shatters.
 32.0s [CLIP ENDS]
```

---

## 4. How it works

```
 episode scripts (.txt)
        │
        ▼
 ┌─────────────────────┐  1. PARSE  (plain code, no AI)
 │ parse.py            │  every script line → ID (E01-L006), type, speaker, start/end seconds
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐  2. FIND  (AI "finder", 5 episodes per call)
 │ pipeline.find()     │  2–3 promo candidates per episode + short "key moments",
 │                     │  pointed at by LINE IDs only, never by timestamps
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐  3. CHECK  (plain code)
 │ resolve_segment()   │  line IDs → exact times · never cut mid-sentence · finish the camera shot
 │ check_rules()       │  20–60 s (trim if longer) · max 3 pieces · pieces ≥3 s · hook in first 3 s?
 └─────────┬───────────┘  anything invalid is rejected and logged with the reason
           ▼
 ┌─────────────────────┐  4. JUDGE  (a DIFFERENT AI model, the "judge")
 │ pipeline.judge()    │  sees only the clip, times from 0 s, as a stranger would
 │                     │  scores hook / cliffhanger / standalone 1–5 and flags spoilers
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐  5. RE-CUT  (AI + code)
 │ pipeline.recut()    │  clips that end on a resolved beat get an earlier end line,
 │                     │  are re-judged, and replace the original only if they score higher
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐  6. MONTAGE  (AI)
 │ pipeline.montages() │  joins 2–4 key moments from different episodes into one arc → judged too
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐  7. RANK  (code + 1 AI call)
 │ score() · rank()    │  score = 0.4×hook + 0.3×cliffhanger + 0.3×standalone − penalties + bonus
 │ head_to_head()      │  remove near-duplicates · AI compares the top 20 side by side
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐  8. WRITE
 │ write() · report()  │  report.md · promos_ranked.csv / .json · rejected / duplicates / recuts
 └─────────────────────┘  run_meta.json · raw/ (every unedited AI reply)
```

### The steps in plain words

1. **Parse (code).** Each script is split into numbered lines. Each line has an ID like `E01-L006` (episode 1, line 6), a type (dialogue, camera, action, sound), a speaker and exact start and end times in seconds. No AI is involved, so this step is exact and instant.

2. **Find (AI finder).** The AI reads 5 episodes at a time, along with the one-line summaries of the story so far. For every episode it proposes 2–3 promo candidates. It says *which lines* to start and end on (`start at E01-L006, end at E01-L020`), never times. It also lists short "key moments" that can later be joined into montages.

3. **Check (code).** The code turns line IDs into exact times and cleans up the cut:
   - it never starts or ends in the middle of someone's sentence;
   - it lets a camera shot finish if it ends within 1.5 s.

   It then enforces the rules. A clip over 60 s is trimmed back to the last line that fits. Clips under 20 s, with more than 3 pieces or with pieces in the wrong order are rejected, and every rejection is written down with its reason. It also checks whether anything is said or done in the first 3 seconds.

4. **Judge (a different AI model).** A second model, deliberately not the one that picked the clips, plays a skeptical ad reviewer. It sees **only the clip's own lines**, with times counted from 0 s, exactly as a stranger would. It scores hook, cliffhanger and "makes sense alone" from 1 to 5, and says whether the clip spoils a twist. For the spoiler check only, it is told the show's premise: the title plus the first 3 episode summaries.

5. **Re-cut (AI + code).** The most common mistake is running one line past the cliffhanger into a resolution, such as an answer, an acceptance or a comforting line. For every clip the judge says ends flat, the finder picks an earlier, unresolved line to end on. The code rebuilds the cut and the judge scores it again. The new version is kept **only if it scores higher**.

6. **Montage (AI).** From all the key moments, the AI builds a few cross-episode montages, for example *humiliated at the altar → a scarred stranger steps in → he proposes*. They go through the same checks and the same judge.

7. **Rank (code, then 1 AI call).**
   - Each promo gets a score: `0.4×hook + 0.3×cliffhanger + 0.3×standalone`.
   - Penalties: −0.5 if nothing happens in the first 3 s, −1.0 for a spoiler, −0.5 if the cliffhanger scored 2 or less.
   - Bonus: +0.25 for clips from episodes 1–10, since ad viewers start at Episode 1.
   - Clips that overlap a better clip by more than 50 % are removed as duplicates.
   - Many clips tie on perfect scores, so the judge compares the **top 20 side by side** in one call and orders them.

8. **Write.** Everything is saved: a readable report, CSV and JSON lists, what was rejected and why, what was re-cut, run statistics, and every unedited AI reply.

The report starts with **"Post these first"**: the best spoiler-free promos, at most one per episode so the first ads aren't all the same scene. Then come **"Hold for later"** (strong clips that spoil a twist) and **"Where the tool picks badly"** (an honest summary of its own weak spots).

### Experimental extras: flash-forwards, drop-off curves, ad packages

`promo_finder/extras.py` runs after ranking. It is on by default; turn it off with `--no-extras`. All three are AI *predictions*, and the report labels them that way.

| Extra | What it does | Why |
|---|---|---|
| **Flash-forward hooks** | For the 15 best single-scene promos, the AI picks the clip's most shocking 1–4 s line (never the final cliffhanger). Code puts it first, then cuts back to the scene's start. The judge scores the new version, which is kept only if it beats the plain one. IDs end in `f`. | Many microdrama ads open this way. It fixes slow openings without changing the scene. |
| **Predicted drop-off curve** | For the top 20, the judge estimates how many of 100 scrollers are still watching after every 3 s, and names the moment most people leave. | It shows *where* a promo loses people, not just a 1–5 score, and gives the app a chart per promo. |
| **Ad package** | For the 10 shortlisted promos: on-screen text for the first 3 s, a second version to A/B test, a caption, hashtags, a Hinglish version and the best-fit audience. Also an `.edl` cut list per top-20 promo. | Turns the shortlist into something marketing can post. The `.edl` imports into Premiere, DaVinci Resolve or Final Cut; link each `EP0xx` reel to that episode's video and the promo is already cut. |

---

## 5. Why it is built this way

| Decision | Why |
|---|---|
| **The AI names line IDs; code computes the times** | AI models often get numbers slightly wrong. Here the AI can only point at lines that exist; an invented ID is rejected. Every timestamp in the output comes straight from the script. |
| **A different model judges the picks** | A model grading its own work is too generous. The judge also sees only the clip, the way a real viewer would. |
| **Rules are enforced in code, not asked of the AI** | "20–60 s", "don't cut mid-sentence" and "max 3 pieces" are facts, not opinions. Code checks them every time. |
| **Re-cut instead of reject** | A good scene with a flat ending is usually one or two lines too long. Fixing the end keeps the scene. |
| **Head-to-head ranking for the top 20** | Scored separately, about a quarter of clips get a perfect 5/5/5. Comparing them side by side separates them. |
| **5 episodes per AI call** | The free Gemini tier allows 20 requests per model per day. Batching makes a 50-episode show fit (29 calls), while still getting 2–3 candidates per episode. |
| **The model is a setting, not code** | Any OpenAI-compatible API works: Gemini, Groq, OpenRouter, OpenAI, Claude, a local Ollama model, or a company gateway. Swap with one line in `.env`. |
| **Everything is cached and logged** | Re-runs are free, an interrupted run resumes, and every decision can be traced back to the exact AI reply in `raw/`. |
| **Fails safely** | If a model is overloaded the tool retries, then falls back to a backup model. If the daily quota runs out, it stops cleanly and still writes everything finished so far. |

---

## 6. What each file does

```
kuku-technologies/
├── run.py                  command-line entry point
├── app.py                  web app (Streamlit)
├── evaluate.py             compare the tool's picks with hand-picked reference promos
├── promo_finder/           the tool itself
│   ├── parse.py            reads scripts into timestamped lines (no AI)
│   ├── llm.py              talks to any AI provider: presets, retries, fallback, cache
│   ├── prompts.py          every instruction sent to the AI, in one place
│   ├── pipeline.py         the 8 steps: find, check, judge, re-cut, montage, rank, write
│   └── extras.py           experimental: flash-forward hooks, drop-off curves, ad packages, .edl files
├── tests/
│   ├── test_core.py        39 offline tests (no API key needed)
│   ├── judge_check.py      live test: does the judge notice when a clip is made worse?
│   └── fixtures/new_show/  a 2-episode show written for testing, in Kuku's format
├── docs/
│   ├── how-it-works.html   visual explainer: open in a browser
│   ├── Promo_Finder_Solution.pdf   2-page solution write-up
│   └── solution.html       source of the PDF
├── outputs/
│   ├── final/              the submitted run on all 50 episodes
│   └── new-show-final/     the run on the test show
├── gold_draft.json         10 reference promos for episodes 1, 5, 12, 20, 33, 47 (used by evaluate.py)
├── .env.example            template for your settings and keys
└── requirements.txt        dependencies: openai (client library), streamlit
```

### In more detail

**`run.py`**: the command line. It reads your settings from `.env`, builds the finder and judge AI clients, runs the pipeline and prints the top of the report. All options are listed in [section 7](#7-settings-and-command-line-options).

**`app.py`**: a Streamlit web page with two tabs:
- *New run*: upload `.txt` scripts or point at a folder, choose the model, and press *Find promos*. You see each step's progress live.
- *Past runs*: browse any earlier run as promo cards. Each card shows the cut list, hook, cliffhanger, the judge's scores and a *Clip script* you can expand. CSV, JSON and report downloads are on the same page.

**`promo_finder/parse.py`**: turns a script into data, with no AI.
- `parse_text()` reads one script into an `Episode` holding a list of `Line`s, each with an ID, type, speaker, tone, start and end.
- `load_episodes()` reads a whole folder and sorts it by episode number.
- `render_for_llm()` writes an episode in the compact, line-numbered form the AI reads:
  `E01-L009 [00:11.00-00:14.00] VANESSA (Cruel): "Oops. Now you finally match the help."`

**`promo_finder/llm.py`**: one small client for every AI provider.
- `PRESETS` holds the base URL, key name, default model and pacing for gemini, groq, openrouter, openai, anthropic and ollama.
- `LLMConfig.from_env()` reads `LLM_*` settings for the finder and `JUDGE_*` settings for the judge.
- `LLM.chat_json()` sends a prompt and returns parsed JSON. It checks the disk cache first. It retries on overload with growing waits and falls back to backup models. It re-asks if the reply isn't valid JSON, and raises `QuotaExhausted` when the daily limit is used up.

**`promo_finder/prompts.py`**: every instruction the AI receives, kept in one file so they are easy to read and tune:
- the promo definition;
- the finder prompt (`FINDER_*`);
- the judge prompt (`JUDGE_*`);
- the re-cut prompt (`RECUT_*`);
- the montage prompt (`MONTAGE_*`);
- the head-to-head prompt (`RANK_*`).

`PROMPT_VERSION` is saved with every run.

**`promo_finder/pipeline.py`**: the steps from [section 4](#4-how-it-works).

| Part | Job |
|---|---|
| `Segment`, `Promo` | one cut piece (episode, start, end, line IDs); one promo (pieces, hook, cliffhanger, why, judge scores, flags) |
| `resolve_segment()` | line IDs → clean start and end times (no mid-sentence cuts, finish the camera shot) |
| `check_rules()`, `trim_to_fit()` | length, order, number of pieces, hook in the first 3 s; trims clips over 60 s |
| `clip_script()` | the clip as a viewer meets it, with times from 0 s (what the judge reads) |
| `score()`, `overlap_ratio()` | the ranking formula; how much two promos overlap |
| `Run.find()` | step 2, in batches of 5 episodes |
| `Run.judge()` | step 4, 20 clips per call |
| `Run.recut()` | step 5 |
| `Run.montages()` | step 6 |
| `Run.rank()`, `Run.head_to_head()` | step 7 |
| `Run.write()`, `Run.report()`, `Run.weak_spots()` | step 8: all output files and the report sections |
| `Run.execute()` | runs every step in order and returns the output folder |

**`promo_finder/extras.py`**: the experimental extras described in [section 4](#experimental-extras-flash-forwards-drop-off-curves-ad-packages).
- `flash_forward()` builds and judges the teaser-first versions.
- `retention()` and `clean_curve()` handle the drop-off curves.
- `ad_package()` writes the ad copy, and `edl()` writes the cut lists for editing software.
- `run_extras()` is called by `Run.execute()`.

**`evaluate.py`**: compares a run with a list of hand-picked promos (`gold_draft.json`). A reference pick counts as **found** if a tool promo covers at least half of it. It prints recall (how many reference picks the tool found) and precision (how many tool promos in those episodes overlap a reference pick).

**`tests/test_core.py`**: 39 tests that run in about 2 seconds with no key and no internet. They cover:
- parsing odd formats, mid-sentence cut snapping, every rule, and trimming;
- a fake AI that invents line IDs, returns broken JSON and skips clips;
- model fallback, stopping on quota, batching, the re-cut step, head-to-head ranking, and the separate judge;
- flash-forward rules, drop-off curve cleaning, `.edl` timecodes, and the extras end to end.

**`tests/judge_check.py`**: a live test of the judge itself. For the top promos it makes two worse copies: one starts 12 s earlier on setup, one ends early before the cliffhanger. All copies go to the judge under hidden IDs. A useful judge should score the originals higher.

---

## 7. Settings and command-line options

### `.env` settings

| Setting | Example | What it does |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | Preset for the finder: `gemini`, `groq`, `openrouter`, `openai`, `anthropic`, `ollama` |
| `GEMINI_API_KEY` | `AIza…` | Key for the preset (or `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) |
| `LLM_MODEL` | `gemini-3.5-flash` | Finder model (default: the preset's model) |
| `LLM_FALLBACK_MODELS` | `gemini-3.6-flash` | Models to try if the main one is overloaded or out of quota |
| `JUDGE_MODEL` | `gemini-2.5-flash` | Judge model. Leave it unset to let the finder judge itself (not recommended) |
| `JUDGE_PROVIDER`, `JUDGE_FALLBACK_MODELS` | | Same as above, for the judge |
| `LLM_BASE_URL`, `LLM_API_KEY` | | Point at any other OpenAI-compatible endpoint, for example a company gateway |

### `run.py` options

| Option | Default | What it does |
|---|---|---|
| `--input FOLDER` | required | Folder of episode `.txt` scripts |
| `--episodes 1-5,9` | all | Only these episodes (good for a quick test) |
| `--provider`, `--model`, `--base-url` | from `.env` | Swap the finder model for this run |
| `--judge-provider`, `--judge-model` | from `.env` | Swap the judge model for this run |
| `--same-judge` | off | Let the finder model judge its own picks |
| `--no-extras` | off | Skip flash-forwards, drop-off curves and ad packages (saves about 5 AI calls) |
| `--max-promos N` | 3 | Candidates per episode |
| `--max-montages N` | 5 | Cross-episode montages (0 = none) |
| `--batch N` | 5 | Episodes per finder call (fewer calls fit free tiers) |
| `--judge-batch N` | 20 | Clips per judge call |
| `--no-cache` | off | Ignore saved replies and ask the AI again |
| `--temperature T` | 0.3 | AI randomness |
| `--out FOLDER` | `outputs` | Where run folders go |
| `--show NAME` | folder name | Show name used in prompts and the report |

Examples:

```bash
python run.py --input "data/<show>" --episodes 1-5                          # quick test on 5 episodes
python run.py --input "data/<show>" --provider groq                         # finder on Groq's free Llama
python run.py --input "data/<show>" --judge-model gemini-3.6-flash          # a different judge
python run.py --input "data/<show>" --provider ollama --model qwen2.5:7b    # fully local, no key
```

---

## 8. Output files

Every run writes a folder `outputs/run-<date>-<time>/`:

| File | What's in it |
|---|---|
| `report.md` | **Start here.** "Post these first", "Hold for later (spoilers)", "Where the tool picks badly", then every promo with its cut list, hook, cliffhanger, why, judge verdict and flags, and finally the rejections |
| `promos_ranked.csv` | One row per promo: rank, ID, type, score, duration, cut list, hook, cliffhanger, why, judge scores, critique, flags, predicted watch-through, on-screen text, caption. Opens in Excel or Sheets |
| `promos_ranked.json` | The same, with full detail per piece (seconds, line IDs, opening and closing lines) |
| `rejected.json` | Candidates that broke a rule, with the reason (too short, invented line ID, …) |
| `duplicates.json` | Promos dropped because a better one covered the same moment |
| `recuts.json` | Each re-cut that was kept: the original and the new version, side by side |
| `ads.json` | Ad package per shortlisted promo: on-screen text A and B, caption, hashtags, Hinglish, audience |
| `edl/` | One `.edl` cut list per top-20 promo, for Premiere, DaVinci Resolve or Final Cut |
| `run_meta.json` | Models used, prompt version, number of AI calls, tokens, time, counts, errors |
| `raw/` | Every unedited AI reply, for full traceability |
| `judge_check.json` | Written by `tests/judge_check.py`, if run |

**Promo IDs:** `E12-P1` is episode 12, candidate 1. An ID ending in `r` (`E03-P1r`) is a re-cut version, and one ending in `f` (`E12-P1f`) is a flash-forward version. `X-1` is montage 1.

---

## 9. Testing and checking quality

```bash
python -m pytest -q                                              # 39 offline tests, ~2 s, no key
python evaluate.py outputs/final/run-<time> gold_draft.json      # compare with reference picks
python tests/judge_check.py outputs/final/run-<time> --top 12    # does the judge notice worse cuts? (uses the API)
python run.py --input tests/fixtures/new_show                    # does it work on a show it has never seen?
python run.py --input "data/<show>" --episodes 1-10 --no-cache   # consistency: run again fresh and compare
```

To score against your own picks, write a `gold.json` like this and run `evaluate.py` with it:

```json

