# Resume Agent

A LangGraph workflow that rewrites a resume to improve its ATS keyword score
for a given job description. It loops through update, score, and review nodes.

## Setup
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your API keys
3. Add `resume.pdf` and `job_description.txt` to the project folder
4. Run `python main.py`

## Notes
Model names on free tiers change often. If you get a 404, update `MODELS` in `llm_setup.py`.