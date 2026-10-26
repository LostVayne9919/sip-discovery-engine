import json, time
import pandas as pd
import altair as alt
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="SIP Discovery Engine", page_icon="🔎", layout="wide")

# ---------- config ----------
MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash"]
DROP = ["passive lapse", "active stop", "pause not resumed"]
COLORS = {"passive lapse": "#0072B2", "active stop": "#E69F00", "pause not resumed": "#009E73"}
COLOR_SCALE = alt.Scale(domain=list(COLORS), range=list(COLORS.values()))

@st.cache_data
def load():
    d = pd.read_csv("engine_data.csv", keep_default_na=False)
    d["agree"] = d["agree"].astype(str).str.lower().eq("true")
    return d

df = load()
rel = df[df.relevant == "yes"]

@st.cache_resource
def client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def ask(prompt, json_out=False):
    cfg = types.GenerateContentConfig(temperature=0,
          response_mime_type="application/json" if json_out else None)
    last = ""
    for m in MODELS:
        for _ in range(2):
            try:
                r = client().models.generate_content(model=m, contents=prompt, config=cfg)
                return r.text
            except Exception as e:
                last = str(e)[:80]; time.sleep(3)
    raise RuntimeError(last)

# ---------- prompts (same as the batch pipeline) ----------
TAG_PROMPT = """You are a research analyst tagging a public post about SIPs (Systematic Investment Plans)
in Indian mutual funds. Text may be English, Hindi or Hinglish. Use ONLY what the text states; never guess.
Return ONE JSON object with keys:
relevant (yes/no: is it the writer's own SIP/mutual-fund experience, doubt or app issue),
investor_type (first-time|experienced|unknown),
journey_month (0-1|2-3|4-6|7-12|12+|unknown),
start_reason (goal|tax|peer|family|finfluencer|employer|unknown),
trigger (low/negative returns|cash crunch|income change|autopay-mandate issue|app UX|confusion|peer-news|need money|distrust|none),
confusion_topic (returns-XIRR|NAV-units|autopay|tax-exit load|pause-cancel process|none),
outside_app (friend-family|Reddit|YouTube|advisor|customer care|none),
income_type (salaried|variable|student|unknown),
unmet_need (max 15 words or "none"),
evidence_quote (exact words from the text, max 20 words, or ""),
branch (active stop|pause not resumed|passive lapse|continuing|n/a).
Rules: advice or opinions to others = relevant "no". Stopping one fund or switching platform while still
investing = "continuing". An instalment failing or being missed without the writer deciding = "passive lapse".
TEXT:
"""
BRANCH_PROMPT = """Classify this post about an Indian mutual-fund SIP. Answer IN ORDER, stop at the first YES:
Q1 Advice to others, market opinion, or not about the writer's own SIP? -> "n/a"
Q2 Still setting up (mandate/autopay/bank) with no instalment yet? -> "n/a"
Q3 An instalment FAILED or was MISSED without the writer choosing it? -> "passive lapse"
   (if they then paid manually and keep investing -> "continuing")
Q4 Writer says they ALREADY stopped/cancelled/redeemed their SIP? -> "active stop" (threats don't count)
Q5 Writer chose to pause/skip, or tried and the app blocked it? -> "pause not resumed"
Q6 Otherwise investing -> "continuing"; unclear -> "n/a"
Examples: "Stopped SIP in one fund, moved it to flexi-cap" -> continuing.
"My SIP was not debited this month even though autopay is on" -> passive lapse.
"Cancelled all my SIPs last month" -> active stop. "I will close my account" -> n/a.
Return ONLY JSON: {"branch": "..."}
TEXT:
"""

# ---------- header ----------
st.title("🔎 SIP Continuation Discovery Engine")
st.caption("Why do first-time investors' SIPs stop within 12 months? An AI pipeline over "
           f"{len(df):,} public posts (Play Store, Reddit, YouTube). "
           "Research tool only. **Not investment advice.** No usernames stored.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Findings", "🏷️ Tag a review", "💬 Ask the data", "🧪 Method & quality"])

# ---------- TAB 1: findings ----------
with tab1:
    c1, c2, c3 = st.columns(3)
    src = c1.multiselect("Source", sorted(df.source.unique()), default=sorted(df.source.unique()))
    app_f = c2.selectbox("App (Play Store)", ["All"] + sorted(df[df.source == "Play Store"].source_detail.unique()))
    hi_only = c3.toggle("High-confidence only (two prompts agree)", value=False)

    f = rel[rel.source.isin(src)]
    if app_f != "All":
        f = f[(f.source != "Play Store") | (f.source_detail == app_f)]
    d = f[f.branch.isin(DROP)]
    if hi_only:
        d = d[d.agree]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Posts analysed", f"{len(df):,}")
    k2.metric("Relevant", f"{len(f):,}")
    k3.metric("Drop-off stories", f"{len(d):,}")
    pl = (d.branch == "passive lapse").mean() if len(d) else 0
    k4.metric("Passive-lapse share", f"{pl:.0%}")

    if len(d):
        st.subheader("How SIPs break: drop-off branch")
        b = d.branch.value_counts().rename_axis("branch").reset_index(name="count")
        b["share"] = b["count"] / b["count"].sum()
        bars = alt.Chart(b).mark_bar(cornerRadiusEnd=4, size=22).encode(
            y=alt.Y("branch:N", sort="-x", title=None),
            x=alt.X("count:Q", title="Drop-off stories"),
            color=alt.Color("branch:N", scale=COLOR_SCALE, legend=None),
            tooltip=["branch", "count", alt.Tooltip("share:Q", format=".0%")]).properties(height=130)
        labels = bars.mark_text(align="left", dx=6, color="#334155").encode(
            text=alt.Text("share:Q", format=".0%"), color=alt.value("#334155"))
        st.altair_chart(bars + labels, width='stretch')

        st.subheader("What triggers each branch")
        t = d.groupby(["trigger", "branch"]).size().reset_index(name="count")
        t = t[t.trigger != "none"]
        st.altair_chart(alt.Chart(t).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("trigger:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("count:Q", title="Drop-off stories"),
            color=alt.Color("branch:N", scale=COLOR_SCALE, legend=alt.Legend(orient="top", title=None)),
            yOffset="branch:N",
            tooltip=["trigger", "branch", "count"]), width='stretch')

        st.subheader("Branch share by source (where people talk shapes what we see)")
        st.dataframe(pd.crosstab(d.source, d.branch, normalize="index").round(2), width='stretch')

        st.subheader("First-time investors only")
        st.dataframe(d[d.investor_type == "first-time"].branch.value_counts().rename("stories"))

        st.subheader("Evidence quotes")
        pick = st.selectbox("Branch", DROP)
        q = d[(d.branch == pick) & (d.evidence_quote != "")][["uid", "source", "evidence_quote"]].head(15)
        st.dataframe(q, width='stretch', hide_index=True)

    st.subheader("Confusion inside the app")
    st.dataframe(f[f.confusion_topic != "none"].confusion_topic.value_counts().rename("posts"))
    st.subheader("What people do outside the app when unsure")
    st.dataframe(f[~f.outside_app.isin(["none", "other"])].outside_app.value_counts().rename("posts"))

# ---------- TAB 2: tag a review ----------
with tab2:
    st.write("Paste any review or comment about a SIP. The engine tags it with the same prompts used on "
             "the full dataset, then runs a **second, independent branch check**. "
             "If the two disagree, it says so instead of guessing.")
    ex = "Autopay failed last month and the app never told me. I only found out when I checked. Haven't restarted."
    txt = st.text_area("Review text", value=ex, height=120)
    if st.button("Tag it", type="primary") and txt.strip():
        with st.spinner("Tagging..."):
            try:
                tags = json.loads(ask(TAG_PROMPT + txt[:1500], json_out=True))
                b2 = json.loads(ask(BRANCH_PROMPT + txt[:1500], json_out=True)).get("branch", "n/a")
            except Exception as e:
                st.error(f"AI is busy right now (free tier). Please retry in a minute. ({e})")
                st.stop()
        agree = tags.get("branch") == b2
        if agree:
            st.success(f"Branch: **{b2}** ✅ both checks agree")
        else:
            st.warning(f"⚠️ Uncertain branch: tagger says **{tags.get('branch')}**, decision-tree says **{b2}**. "
                       "A human should review this one.")
        quote = tags.get("evidence_quote", "")
        if quote and quote.lower() not in txt.lower():
            st.info("Evidence quote was not found word-for-word in the text, so it was removed.")
            tags["evidence_quote"] = ""
        st.dataframe(pd.DataFrame(tags.items(), columns=["tag", "value"]), hide_index=True, width='stretch')

# ---------- TAB 3: ask the data ----------
with tab3:
    st.write("Ask a question about the dataset. Answers use **only** counts and quotes from the tagged data, "
             "with post IDs so you can check them. If the data can't answer, the engine says so.")
    qs = ["Why do passive lapses happen?", "What confuses first-time investors inside the app?",
          "How does distrust lead to stopping a SIP?", "Do low returns cause people to stop?"]
    qsel = st.selectbox("Try a question", qs)
    question = st.text_input("Or type your own", value=qsel)
    if st.button("Ask", type="primary") and question.strip():
        words = [w for w in question.lower().split() if len(w) > 3]
        sc = rel.text.str.lower().apply(lambda t: sum(w in t for w in words))
        hits = rel.assign(score=sc).sort_values("score", ascending=False)
        hits = hits[hits.score > 0].head(40)
        stats = {
            "drop_off_branch_counts": rel[rel.branch.isin(DROP)].branch.value_counts().to_dict(),
            "trigger_by_branch": pd.crosstab(rel[rel.branch.isin(DROP)].trigger,
                                             rel[rel.branch.isin(DROP)].branch).to_dict(),
            "first_time_branch": rel[(rel.investor_type == "first-time") & rel.branch.isin(DROP)].branch.value_counts().to_dict(),
            "confusion_topics": rel.confusion_topic.value_counts().to_dict(),
        }
        ctx = hits[["uid", "source", "branch", "trigger", "evidence_quote"]].to_dict("records")
        prompt = f"""You answer questions about a research dataset of public posts on SIP continuation in India.
Use ONLY the STATS and POSTS below. Cite post IDs like [R01234]. Give counts where possible.
If the data does not answer the question, reply exactly: "The dataset does not answer this."
Never give investment advice or recommend funds. Max 150 words.
STATS: {json.dumps(stats)}
POSTS: {json.dumps(ctx, ensure_ascii=False)}
QUESTION: {question}"""
        with st.spinner("Searching 5,400+ relevant posts..."):
            try:
                ans = ask(prompt)
            except Exception as e:
                st.error(f"AI is busy right now (free tier). Please retry in a minute. ({e})"); st.stop()
        cited = set(__import__("re").findall(r"R\d{5}", ans))
        bad = cited - set(rel.uid)
        if bad:
            st.warning(f"Removed answer: it cited IDs that don't exist ({', '.join(bad)}).")
        else:
            st.markdown(ans)
            if cited:
                st.caption("Cited posts")
                st.dataframe(rel[rel.uid.isin(cited)][["uid", "source", "branch", "text"]],
                             hide_index=True, width='stretch')
        st.caption("Not investment advice. For official SIP information see amfiindia.com and sebi.gov.in.")

# ---------- TAB 4: method ----------
with tab4:
    st.subheader("Pipeline")
    st.markdown("""
1. **Collect:** Play Store reviews (5 investing apps), Reddit (5 Indian investing subreddits), YouTube comments (24 videos, English + Hinglish)
2. **Filter:** SIP keywords, minimum length, IPO-mandate noise removed
3. **Tag:** Gemini Flash-Lite, 12 structured fields per post (branch, trigger, confusion, outside-app action, segment, unmet need, evidence quote)
4. **Second pass:** branch re-checked with a step-by-step decision tree; posts where both passes agree = high confidence
5. **Analyse:** branch × trigger × source × investor type
""")
    st.subheader("Volume")
    st.dataframe(df.source.value_counts().rename("posts"))
    st.subheader("Quality checks against human labels")
    st.dataframe(pd.DataFrame({
        "Prompt version": ["v1", "v2 (rules)", "v2 on fresh holdout", "v3 decision tree", "v3b + examples", "v2 & v3b agree"],
        "Branch accuracy": ["68%", "100%", "40%", "71%", "88%", "100%"],
        "Test set": ["19 golden", "19 golden", "15 holdout", "34 dev", "34 dev", "23 of 34 dev"],
        "Note": ["base prompt", "overfit to golden set", "exposed the overfit", "fixed order of questions",
                 "made-up examples", "used for high-confidence view"]}), hide_index=True, width='stretch')
    st.subheader("Known limits")
    st.markdown("""
- Reviews rarely say *when* a SIP stopped (journey month unknown for most posts) or the writer's income type. Interviews fill this gap.
- Play Store over-represents app failures; Reddit over-represents experienced investors. Results are shown by source.
- 50 posts (0.7%) could not be tagged after retries and are excluded.
""")
