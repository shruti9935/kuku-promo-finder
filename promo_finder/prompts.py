"""Prompts. Kept in one file so they are easy to read, version and tune."""

PROMPT_VERSION = "v3"

PROMO_DEFINITION = """A PROMO is a 20-60 second clip cut from the show and posted on TikTok / Instagram Reels /
YouTube Shorts to pull NEW viewers in (think ReelShort and DramaBox ads). A good promo:
1. HOOK: opens on something in the FIRST 3 SECONDS that stops a scrolling thumb: a slap, a
   reveal, a shocking or insulting line, a humiliation, a threat, a kiss, a betrayal. It must not
   open on setup, establishing shots or calm small talk.
2. CLIFFHANGER: cuts at an unresolved moment (a question unanswered, a threat hanging, a
   reveal about to land, a door opening) so the viewer needs to know what happens next.
3. STANDS ALONE: makes sense to a stranger who has never seen the show. Clear conflict, clear
   underdog/villain, no reliance on names or backstory the clip doesn't show.
4. NO BIG SPOILERS: don't give away the show's central secret or late twists; tease them.
Most promos are ONE continuous stretch. A promo may join 2-3 stretches (each at least 4 seconds)
if that removes boring setup or gets to the cliffhanger faster; stretches must be in story order."""

FINDER_SYSTEM = f"""You are a senior promo editor for vertical microdrama apps. You pick ad clips that
convert scrolling strangers into viewers.

{PROMO_DEFINITION}

You read a timestamped, line-numbered screenplay. Every line has an ID like L042 and a time
range. Point at moments ONLY by line IDs that exist in the script. Never invent lines.
Use the timestamps to make sure each promo totals 20-60 seconds (aim for 25-45).
Reply with one JSON object and nothing else."""

FINDER_USER = """SHOW: {show}
STORY SO FAR (episode loglines, for your context only; viewers of the promo have NOT seen these):
{story}

{episodes}

TASK
Work through EACH episode above separately. Do not skip any episode.
A) For EVERY episode, give {max_promos} promo candidates (at least 2, even in a quiet episode: give the best
   moments it has, and later checks drop the weak ones). Candidates should be different scenes. Choose segments by line IDs: the promo starts at the START of `start_line` and ends at the END
   of `end_line`. All line IDs of one promo come from ONE episode. The first line of the first segment must
   BE the hook (or begin no more than 3 seconds before it). The last line must be the unresolved moment.
B) For each episode, list up to 3 short KEY MOMENTS (3-20 s each) that could be building blocks in a
   cross-episode montage: strong hooks, big reveals, or strong cliffhanger beats.

JSON format:
{{
  "promos": [
    {{
      "episode": 3,
      "title": "short working title",
      "segments": [{{"start_line": "E03-L012", "end_line": "E03-L040"}}],
      "hook": "what happens in the first 3 seconds",
      "cliffhanger": "the unresolved moment it cuts on",
      "why": "one line on why a stranger would stop and want more",
      "context_needed": "anything a stranger would be confused by, or 'none'"
    }}
  ],
  "key_moments": [
    {{"episode": 3, "start_line": "E03-L050", "end_line": "E03-L058", "type": "hook|reveal|cliffhanger", "what": "one line"}}
  ]
}}"""

EPISODE_BLOCK = """=== EPISODE {ep} ===
Logline: {logline}
Length: {duration}
SCRIPT (line ID, [start-end], speaker or type, text):
{script}"""

JUDGE_SYSTEM = """You are a tough, skeptical social-media ad reviewer. You see ONLY the clip's script, exactly
as a stranger scrolling TikTok would experience it, with no knowledge of the show. Score honestly;
most clips are average. A 5 is rare: give it only to a clip that would beat most ads in the feed.
Use the whole 1-5 range across a batch; if every clip gets 4-5 your scores are useless for ranking.
Reply with one JSON object and nothing else."""

JUDGE_USER = """Rate each promo clip below. Each clip is shown as the exact screenplay lines it contains, with
times relative to the start of the clip (0.0 s = first frame the viewer sees).

Scores 1-5:
- hook: does something in the first 3 seconds (0-3 s) make a scroller stop? 5 = slap/reveal/shocking
  line right at 0 s; 3 = mildly intriguing; 1 = setup, silence, establishing shot.
- cliffhanger: does the clip END on an unresolved moment that makes you need the next scene?
  5 = cut right before a reveal/answer/consequence; 1 = ends on a resolved or neutral beat.
- standalone: would a stranger understand who is who and what's at stake? 5 = crystal clear;
  1 = confusing without prior episodes.
- spoiler: true only if the clip gives away a major twist BEYOND the premise below (what the show's title
  and first episodes already tell every new viewer). A clip that only shows the premise is not a spoiler.

SHOW PREMISE (for the spoiler check only; score everything else as a stranger):
{premise}
Judge the 0-3 s window from the lines that START before 3.0 s. A camera line alone is a weak hook
unless it describes something shocking happening on screen.

CLIPS:
{clips}

JSON format:
{{"ratings": [{{"id": "<clip id>", "hook": 1-5, "cliffhanger": 1-5, "standalone": 1-5,
  "spoiler": true/false, "critique": "one line: the main weakness, or why it works"}}]}}"""

MONTAGE_SYSTEM = f"""You are a senior promo editor for vertical microdrama apps.

{PROMO_DEFINITION}

You build MONTAGE promos that join 2-4 short moments from DIFFERENT episodes into one 20-60 s ad
that tells a simple arc (e.g. humiliation -> secret identity -> revenge is coming). Reply with one
JSON object and nothing else."""

MONTAGE_USER = """SHOW: {show}
STORY (episode loglines):
{story}

AVAILABLE MOMENTS (id, episode, duration, type, description, first line, last line):
{moments}

Build up to {max_montages} montage promos. Each uses 2-4 moments by id, in story order (earlier episode
first), opens on a hook moment, ends on a cliffhanger moment, totals 20-60 seconds, and makes sense to a
stranger. Prefer moments from early episodes so new viewers land on Episode 1 without big spoilers.

JSON format:
{{"montages": [{{"title": "...", "moment_ids": ["M003", "M017"], "hook": "...", "cliffhanger": "...",
  "why": "one line"}}]}}"""

RANK_SYSTEM = JUDGE_SYSTEM

RANK_USER = """HEAD-TO-HEAD. These clips all scored well on their own, so the scores no longer separate them.
Compare them against EACH OTHER and order them from the ad you would spend money on first to the one you
would spend on last. Think of a stranger scrolling: which clip stops the thumb fastest, makes the most
sense with zero context, and leaves the strongest need to see what happens next?

CLIPS:
{clips}

JSON format:
{{"order": ["<best clip id>", "...", "<worst clip id>"], "best_reason": "one line on why the first beats the rest"}}"""

RECUT_SYSTEM = FINDER_SYSTEM

RECUT_USER = """RE-CUT. A reviewer said each clip below ends on a beat that feels RESOLVED (an answer, an
acceptance, a comfort, a calm exit), so a stranger has no reason to watch the next episode.
For each clip, pick the line to END on instead: an EARLIER line where the tension peaks and is still
unresolved (a threat, a question, an accusation, a reveal about to land, a choice not yet made).
Rules: pick only a line ID listed for that clip; the clip must still be at least 20 seconds long (times are
shown from the clip's start); if no earlier line works, return "keep".

CLIPS:
{clips}

JSON format:
{{"recuts": [{{"id": "<clip id>", "end_line": "E05-L031 or keep", "why": "one line"}}]}}"""

# ---------- experimental extras (promo_finder/extras.py) ----------

TEASER_USER = """FLASH-FORWARD. Many top microdrama ads open with a 1-3 second flash of the clip's most shocking
moment (a slap, a reveal, a scream, a threat), then cut back to where the scene starts. For each clip below, pick
the ONE line to flash first: the most arresting single line or action, ideally 1-3 seconds long. It must not be
the clip's final line (that's the cliffhanger; don't spoil it). If the clip already opens on its strongest
moment, return "none".

CLIPS (line ID, time from the clip's start, line):
{clips}

JSON format:
{{"teasers": [{{"id": "<clip id>", "line": "E05-L031 or none", "why": "one line"}}]}}"""

RETENTION_USER = """PREDICTED DROP-OFF. Imagine 100 strangers scrolling a vertical video feed who see each clip below.
For every 3-second window of the clip, estimate how many of the 100 are STILL watching at the end of that window.
Feeds are brutal: a weak opening loses half the audience within 3 seconds, slow stretches lose people steadily,
and a strong hook or rising tension holds them. Numbers must never go up. Be realistic; most ads end with 10-40
still watching. Also name the moment where the most people leave.

CLIPS:
{clips}

JSON format:
{{"curves": [{{"id": "<clip id>", "still_watching": [100-ish at 3 s, ..., at the end], "biggest_drop": "e.g. 12-15 s: two people talk calmly"}}]}}"""

ADCOPY_USER = """AD PACKAGE. For each promo below, write what a performance-marketing team needs to post it.
- overlay: on-screen text shown over the first 3 seconds, at most 8 words, punchy, may use one emoji
- overlay_b: a second overlay to A/B test, with a different angle (e.g. question vs statement)
- caption: the post caption, at most 20 words, ends with a call to watch (e.g. "Watch Episode 1 now")
- hashtags: 4-6 hashtags
- hinglish: the overlay rewritten in natural Hinglish (Hindi in Latin script, mixed with English) for Indian viewers
- audience: the viewer this ad suits best, in a few words
Never reveal the cliffhanger's answer or the show's big twists.

SHOW: {show}

PROMOS:
{clips}

JSON format:
{{"ads": [{{"id": "<promo id>", "overlay": "...", "overlay_b": "...", "caption": "...", "hashtags": ["#..."],
  "hinglish": "...", "audience": "..."}}]}}"""
