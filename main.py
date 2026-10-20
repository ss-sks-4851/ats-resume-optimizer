import re
import sys
import operator
from typing import Annotated, TypedDict

from pypdf import PdfReader
from langgraph.graph import StateGraph, START, END

from llm_setup import get_llm, text_of

# =====================================================================
# 1. MODELS (one per role)
# =====================================================================
analyzer_llm = get_llm("google")                              # keyword extraction
updater_llm  = get_llm("google")                              # rewrites the resume
reviewer_llm = get_llm("groq", model="openai/gpt-oss-120b")   # critic (different model family)


# =====================================================================
# 2. SETTINGS
# =====================================================================
TARGET_SCORE = 85        # stop when score reaches this AND the reviewer approves
MAX_ITERATIONS = 3       # hard stop so the loop can never run forever
REQUIRED_WEIGHT = 0.7    # required keywords count more than preferred ones
PREFERRED_WEIGHT = 0.3
EMPTY_VALUES = {"none", "n/a", "na", "nil", "-", ""}   # placeholders the LLM might return


# =====================================================================
# 3. STATE (shared memory of the graph)
# =====================================================================
class ResumeState(TypedDict):
    resume_path: str
    job_description: str
    original_resume: str      # never modified: the source of truth
    current_resume: str       # the version being improved
    jd_analysis: dict         # {"job_title", "required_keywords", "preferred_keywords"}
    ats_score: float
    missing_keywords: list[str]
    feedback: str             # reviewer's notes for the next update
    approved: bool
    iteration: int
    score_history: Annotated[list[float], operator.add]   # reducer: appends, never overwrites


# =====================================================================
# 4. HELPERS
# =====================================================================
def ask(llm, prompt: str, check=None, tries: int = 3) -> str:
    """Call the LLM until the reply is non-empty and passes `check` (if given).
    On retries we add a nudge so the model doesn't just repeat itself."""
    text = ""
    for attempt in range(1, tries + 1):
        nudge = ""
        if attempt > 1:
            nudge = ("\n\nIMPORTANT: Your previous reply was incomplete or badly "
                     "formatted. Reply again with EVERY required field fully filled in.")
        text = text_of(llm.invoke(prompt + nudge)).strip()
        if text and (check is None or check(text)):
            return text
        print(f"  (bad or incomplete reply, retry {attempt}/{tries})")
    raise ValueError(f"No valid reply after {tries} tries. Last reply:\n{text}")

def grab(text: str, label: str) -> str:
    """Pull the value after 'LABEL:' on its own line."""
    m = re.search(rf"^{label}:\s*(.*)$", text, re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def to_list(s: str) -> list[str]:
    """'a, b, None' -> ['a', 'b']  (drops placeholder values)"""
    items = [k.strip() for k in s.split(",") if k.strip()]
    return [k for k in items if k.lower() not in EMPTY_VALUES]

def keyword_present(keyword: str, text: str) -> bool:
    """Whole-term match, safe for things like 'C++', 'C#', 'Node.js'."""
    pattern = r"(?<![A-Za-z0-9+#])" + re.escape(keyword.lower()) + r"(?![A-Za-z0-9+#])"
    return re.search(pattern, text.lower()) is not None


# =====================================================================
# 5. NODES
# =====================================================================
def parse_resume(state: ResumeState):
    print("[parse_resume] reading PDF...")
    reader = PdfReader(state["resume_path"])
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise ValueError("No text extracted. The PDF may be scanned/image-based.")

    jd = state["job_description"]
    if len(jd.strip()) < 300:
        raise ValueError(f"Job description looks too short ({len(jd)} chars). Is the file complete?")

    return {"original_resume": text, "current_resume": text, "iteration": 0}


def analyze_jd(state: ResumeState):
    print("[analyze_jd] extracting keywords...")

    def is_complete(text: str) -> bool:
        text = text.replace("*", "")
        return bool(grab(text, "TITLE")) and bool(to_list(grab(text, "REQUIRED")))

    text = ask(analyzer_llm, f"""Extract ATS keywords from this job description.
Reply in EXACTLY this format and nothing else:
TITLE: <job title>
REQUIRED: <comma-separated must-have skills and tools>
PREFERRED: <comma-separated nice-to-have skills and tools, or leave blank if none>

Rules:
- Include only specific, named technologies, tools, methods, and certifications
  (e.g. "Python", "Apache Airflow", "dimensional modeling", "CI/CD").
- Do NOT include generic categories or phrases (e.g. "cloud platform",
  "orchestration tool", "distributed processing framework", "data engineering").
- Do NOT include years of experience, degrees, or soft skills.
- Use short terms (1-3 words each), as written in the posting.
- Put items from "Requirements" under REQUIRED and items from
  "Nice to have" or "Preferred" under PREFERRED.

JOB DESCRIPTION:
{state['job_description']}""", check=is_complete).replace("*", "")

    analysis = {
        "job_title": grab(text, "TITLE"),
        "required_keywords": to_list(grab(text, "REQUIRED")),
        "preferred_keywords": to_list(grab(text, "PREFERRED")),
    }
    print(f"           required:  {analysis['required_keywords']}")
    print(f"           preferred: {analysis['preferred_keywords']}")
    return {"jd_analysis": analysis}


def score(state: ResumeState):
    text = state["current_resume"]
    required = state["jd_analysis"]["required_keywords"]
    preferred = state["jd_analysis"]["preferred_keywords"]

    req_hits = [k for k in required if keyword_present(k, text)]
    pref_hits = [k for k in preferred if keyword_present(k, text)]

    req_cov = len(req_hits) / len(required) if required else 1.0
    pref_cov = len(pref_hits) / len(preferred) if preferred else 1.0

    total = round(100 * (REQUIRED_WEIGHT * req_cov + PREFERRED_WEIGHT * pref_cov), 1)
    missing = [k for k in required + preferred if not keyword_present(k, text)]

    print(f"[score] ATS score: {total}  |  missing: {missing}")
    return {"ats_score": total, "missing_keywords": missing, "score_history": [total]}


def update(state: ResumeState):
    n = state["iteration"] + 1
    print(f"[update] rewriting (iteration {n})...")

    prompt = f"""You are an expert resume editor optimizing a resume for a specific job.

JOB TITLE: {state['jd_analysis']['job_title']}
REQUIRED KEYWORDS: {state['jd_analysis']['required_keywords']}
PREFERRED KEYWORDS: {state['jd_analysis']['preferred_keywords']}
KEYWORDS CURRENTLY MISSING: {state['missing_keywords']}

ORIGINAL RESUME (ground truth, do not contradict it):
{state['original_resume']}

CURRENT DRAFT:
{state['current_resume']}

REVIEWER FEEDBACK TO ADDRESS:
{state.get('feedback') or 'None yet. This is the first pass.'}

RULES:
1. NEVER invent experience, skills, employers, dates, metrics, or projects.
   Only use facts present in the ORIGINAL RESUME.
2. Add a missing keyword ONLY if the original resume genuinely supports it.
   Use the job description's wording for things the candidate really did.
3. Remove or shorten projects and skills that are irrelevant to this job.
4. Reorder so the most relevant projects and skills come first.
5. Keep bullets concise, action-oriented, and quantified where the original had numbers.
6. Use simple ATS-friendly structure: plain section headings (Summary, Skills,
   Experience, Projects, Education), no tables, no columns.

Return ONLY the full rewritten resume as plain text. No commentary, no code fences."""

    # sanity check: a real resume rewrite is never tiny
    text = ask(updater_llm, prompt, check=lambda t: len(t) > 200)
    text = re.sub(r"^```[a-z]*\n|\n```$", "", text).strip()   # strip accidental code fences
    return {"current_resume": text, "iteration": n}



def review(state: ResumeState):
    print("[review] critiquing the draft...")

    prompt = f"""You are a strict resume reviewer.

JOB TITLE: {state['jd_analysis']['job_title']}
REQUIRED KEYWORDS: {state['jd_analysis']['required_keywords']}
CURRENT ATS SCORE: {state['ats_score']} (missing: {state['missing_keywords']})

ORIGINAL RESUME:
{state['original_resume']}

REWRITTEN DRAFT:
{state['current_resume']}

Check for: (1) any claims not supported by the original resume, (2) relevance of
projects and skills to the job, (3) weak or vague bullets, (4) missing keywords that
the original legitimately supports, (5) formatting that could confuse an ATS.

Reply in EXACTLY this format:
VERDICT: <write APPROVED or REVISE>
FEEDBACK: <specific, actionable instructions for the next revision. If you found
unsupported claims, name them and say to remove them.>

Choose APPROVED only if there are no unsupported claims and the draft is strong."""

    text = ask(
        reviewer_llm, prompt,
        check=lambda t: "VERDICT:" in t.replace("*", "").upper(),
    ).replace("*", "")

    verdict = grab(text, "VERDICT").upper()
    approved = verdict.startswith("APPROVED")    # anything unclear counts as not approved

    feedback = text.split("FEEDBACK:", 1)[1].strip() if "FEEDBACK:" in text else text
    print(f"         verdict={verdict or '?'}  approved={approved}")
    return {"approved": approved, "feedback": feedback}


# =====================================================================
# 6. ROUTING (decides which node runs next)
# =====================================================================
def route_after_score(state: ResumeState):
    # iteration 0 = we only have the baseline score, so go rewrite.
    # later      = the draft has been scored, so go review it.
    return "update" if state["iteration"] == 0 else "review"


def route_after_review(state: ResumeState):
    if state["approved"] and state["ats_score"] >= TARGET_SCORE:
        return END                      # success
    if state["iteration"] >= MAX_ITERATIONS:
        return END                      # safety cap
    return "update"                     # try again using the feedback


# =====================================================================
# 7. GRAPH
# =====================================================================
g = StateGraph(ResumeState)

g.add_node("parse_resume", parse_resume)
g.add_node("analyze_jd", analyze_jd)
g.add_node("score", score)
g.add_node("update", update)
g.add_node("review", review)

g.add_edge(START, "parse_resume")
g.add_edge("parse_resume", "analyze_jd")
g.add_edge("analyze_jd", "score")
g.add_conditional_edges("score", route_after_score, {"update": "update", "review": "review"})
g.add_edge("update", "score")
g.add_conditional_edges("review", route_after_review, {"update": "update", END: END})

app = g.compile()

# =====================================================================
# 8. RUN
# =====================================================================
if __name__ == "__main__":
    result = app.invoke({
        "resume_path": "resume.pdf",
        "job_description": open("job_description.txt", encoding="utf-8").read(),
        "feedback": "",
        "approved": False,
        "score_history": [],
    })

    print("\nScore history:", result["score_history"])
    print("Approved:", result["approved"])
    print("\n===== FINAL RESUME =====\n")
    print(result["current_resume"])

    with open("final_resume.txt", "w", encoding="utf-8") as f:
        f.write(result["current_resume"])
    print("\nSaved to final_resume.txt")