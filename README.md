# V.I.T.A.L. — Vital Intelligence & Triage Assistance Layer

A structured AI health-intelligence console: symptoms + lifestyle + medical
history → a triage-graded assessment (stable / monitor / emergency) plus a
deterministic wellness score. Built with Flask and Google Gemini.

> Educational project. Not a medical device, not a diagnosis, not a substitute
> for a clinician.

## What makes it more than a chatbot
- **Structured triage** — Gemini returns typed JSON (causes, reasoning, self-care,
  precautions, action, red flags, triage level, confidence), rendered as cards
  rather than a wall of text.
- **Wellness engine** — a transparent, rule-based 0–100 score across Sleep,
  Activity, Nutrition and Substances. No AI, fully explainable, updates live.
- **Persistent profile** — chronic history + lifestyle feed every assessment as
  context; history is saved per user.
- **Country-aware emergency routing** and a conservative, safety-first prompt.

## Run it
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your_new_key"     # Windows: setx GEMINI_API_KEY "..."
python app.py
# open http://localhost:5001
```

## Layout
```
app.py                  Flask app · wellness engine · Gemini triage
templates/login.html    onboarding + chronic history
templates/dashboard.html wellness ring + live lifestyle scoring + history
templates/chatbot.html  the assessment console
static/css/style.css    diagnostic-console design system
```

## Notes
- Uses the current `google-genai` SDK and the `gemini-3.5-flash` model.
- Data persists to `users.json` (fine for a demo; swap for a real DB to scale).
