"""
V.I.T.A.L. — Vital Intelligence & Triage Assistance Layer
A structured AI health-intelligence console built on Flask + Google Gemini.

This is an educational project. It does NOT diagnose. Every assessment is
framed as guidance and always routes serious cases to real medical care.
"""

import os
import json
import uuid
from datetime import datetime, timezone

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, abort
)

# New Google Gen AI SDK (replaces the deprecated google.generativeai)
from google import genai
from google.genai import types

# --------------------------------------------------------------------------- #
#  CONFIG
# --------------------------------------------------------------------------- #
def _load_dotenv(filename=".env"):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(path):
        return
    # utf-8-sig transparently strips a BOM that Notepad adds and that would
    # otherwise corrupt the first variable's name.
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip().lstrip("\ufeff")
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lstrip("\ufeff")
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()

# The key is read from the environment / .env. Never hard-code it in source.
API_KEY = os.environ.get("GEMINI_API_KEY")

MODEL = "gemini-3.5-flash"  # current stable Flash; also aliased as gemini-flash-latest
DATA_FILE = os.path.join(os.path.dirname(__file__), "users.json")

client = genai.Client(api_key=API_KEY) if API_KEY else None

app = Flask(__name__)
app.secret_key = os.environ.get("VITAL_SECRET", "dev-secret-change-me")

EMERGENCY_NUMBERS = {
    "india": "112", "usa": "911", "united states": "911", "uk": "999",
    "united kingdom": "999", "canada": "911", "australia": "000",
    "singapore": "995", "uae": "998", "germany": "112", "france": "112",
}


# --------------------------------------------------------------------------- #
#  STORAGE
# --------------------------------------------------------------------------- #

def load_users():
    if not os.path.exists(DATA_FILE):
        return {}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_users(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# --------------------------------------------------------------------------- #
#  WELLNESS ENGINE  (deterministic, explainable — no AI involved)
#  Turns the lifestyle questionnaire into a 0-100 score across four pillars.
# --------------------------------------------------------------------------- #

SCORE_MAP = {
    "sleep_hours":   {"<5h": 20, "5-6h": 50, "6-7h": 72, "7-8h": 100, "8+h": 85},
    "sleep_quality": {"Poor": 30, "Average": 65, "Good": 100},
    "wake_refreshed": {"Yes": 100, "No": 40},
    "exercise_freq": {"0": 20, "1-2": 55, "3-4": 85, "5+": 100},
    "exercise_type": {"None": 30, "Strength": 85, "Cardio": 85, "Mixed": 100},
    "diet_type":     {"Vegetarian": 82, "Non-Veg": 74, "Vegan": 82},
    "processed_food": {"Low": 100, "Moderate": 58, "High": 25},
    "alcohol":       {"None": 100, "Social": 80, "Weekly": 55, "Heavy": 15},
    "smoking":       {"Never": 100, "Former": 72, "Current": 20},
    "caffeine":      {"0": 88, "1-2 cups": 100, "3-4": 68, "5+": 40},
}


def _avg(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values)) if values else None


def _lookup(field, value):
    return SCORE_MAP.get(field, {}).get(value)


def compute_wellness(lifestyle):
    """Return {overall, pillars:{...}, completeness} from a lifestyle dict."""
    L = lifestyle or {}

    sleep = _avg([
        _lookup("sleep_hours", L.get("sleep_hours")),
        _lookup("sleep_quality", L.get("sleep_quality")),
        _lookup("wake_refreshed", L.get("wake_refreshed")),
    ])
    activity = _avg([
        _lookup("exercise_freq", L.get("exercise_freq")),
        _lookup("exercise_type", L.get("exercise_type")),
    ])

    diet_s = _lookup("diet_type", L.get("diet_type"))
    proc_s = _lookup("processed_food", L.get("processed_food"))
    if diet_s is not None and proc_s is not None:
        nutrition = round(0.4 * diet_s + 0.6 * proc_s)   # processed food weighs more
    else:
        nutrition = diet_s if diet_s is not None else proc_s

    substances = _avg([
        _lookup("alcohol", L.get("alcohol")),
        _lookup("smoking", L.get("smoking")),
        _lookup("caffeine", L.get("caffeine")),
    ])

    pillars = {"Sleep": sleep, "Activity": activity,
               "Nutrition": nutrition, "Substances": substances}
    weights = {"Sleep": 0.30, "Activity": 0.25, "Nutrition": 0.25, "Substances": 0.20}

    present = {k: v for k, v in pillars.items() if v is not None}
    if present:
        total_w = sum(weights[k] for k in present)
        overall = round(sum(pillars[k] * weights[k] for k in present) / total_w)
    else:
        overall = None

    completeness = round(100 * len(present) / len(pillars))
    return {"overall": overall, "pillars": pillars, "completeness": completeness}


# --------------------------------------------------------------------------- #
#  AI TRIAGE  (structured JSON output)
# --------------------------------------------------------------------------- #

TRIAGE_SCHEMA_HINT = """
Return ONLY a JSON object (no markdown, no backticks) with exactly this shape:
{
  "triage_level": "stable" | "monitor" | "emergency",
  "confidence": "low" | "moderate" | "high",
  "summary": "one plain-language sentence",
  "likely_causes": [ {"name": "...", "explanation": "..."} ],
  "reasoning": [ "why the inputs point this way", "..." ],
  "self_care": [ "actionable step", "..." ],
  "precautions": [ "what to avoid / watch for", "..." ],
  "recommended_action": "the single clearest next step",
  "seek_help_if": [ "red-flag symptom", "..." ]
}
"""

SYSTEM_INSTRUCTION = (
    "You are V.I.T.A.L., an AI health-intelligence assistant for an educational "
    "project. You are NOT a doctor and must never state a definitive diagnosis. "
    "Offer possibilities, not verdicts. Be calm, specific and conservative: when "
    "inputs are ambiguous or concerning, err toward a higher triage level and "
    "toward recommending a real clinician. Keep every list item to one short "
    "sentence. Use plain language a teenager could follow."
)


def build_prompt(profile, wellness, symptoms, vitals, facial):
    chronic = profile.get("history", "None recorded")
    lifestyle = profile.get("lifestyle", {})
    ws = wellness.get("overall")
    return f"""{TRIAGE_SCHEMA_HINT}

PATIENT PROFILE
Chronic history: {chronic}
Lifestyle: {json.dumps(lifestyle) if lifestyle else "not provided"}
Wellness score: {ws if ws is not None else "n/a"}/100

CURRENT PRESENTATION
Symptoms: {symptoms or "none stated"}
Vitals: {vitals or "none stated"}
Facial / visible indicators: {facial or "none stated"}

Assess this presentation and respond with the JSON object only."""


def run_triage(prompt):
    """Call Gemini and return a parsed dict, or a safe fallback on any failure."""
    if client is None:
        return _fallback("GEMINI_API_KEY is not set on the server.")
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.4,
            ),
        )
        raw = (resp.text or "").strip()
        # strip stray fences just in case
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[raw.find("{"):]
        data = json.loads(raw)
        return _normalise(data)
    except json.JSONDecodeError:
        return _fallback("The model returned an unreadable response. Please retry.")
    except Exception as e:                       # noqa: BLE001
        print("TRIAGE ERROR:", e)
        return _fallback("The assessment engine is temporarily unavailable.")


def _normalise(d):
    """Guarantee every field exists and triage_level is valid."""
    level = str(d.get("triage_level", "monitor")).lower()
    if level not in ("stable", "monitor", "emergency"):
        level = "monitor"
    return {
        "triage_level": level,
        "confidence": str(d.get("confidence", "moderate")).lower(),
        "summary": d.get("summary", "Assessment generated."),
        "likely_causes": d.get("likely_causes", []) or [],
        "reasoning": d.get("reasoning", []) or [],
        "self_care": d.get("self_care", []) or [],
        "precautions": d.get("precautions", []) or [],
        "recommended_action": d.get("recommended_action", "Monitor your symptoms."),
        "seek_help_if": d.get("seek_help_if", []) or [],
    }


def _fallback(message):
    return {
        "triage_level": "monitor",
        "confidence": "low",
        "summary": message,
        "likely_causes": [],
        "reasoning": [],
        "self_care": [],
        "precautions": [],
        "recommended_action": "If you feel unwell, contact a healthcare professional.",
        "seek_help_if": [],
    }


# --------------------------------------------------------------------------- #
#  ROUTES
# --------------------------------------------------------------------------- #

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        if not name:
            return render_template("login.html", error="Please enter your name.")

        country = (request.form.get("country") or "india").strip().lower()

        history = "\n".join([
            f"Country: {country}",
            f"Heart disease: {request.form.get('q1', 'No')}",
            f"Diabetes: {request.form.get('q2', 'No')}",
            f"High blood pressure: {request.form.get('q3', 'No')}",
            f"Respiratory issues: {request.form.get('q4', 'No')}",
            f"Stress / anxiety: {request.form.get('q5', 'No')}",
            f"Other: {request.form.get('custom_issue') or 'None'}",
        ])

        users = load_users()
        existing = users.get(name, {})
        users[name] = {
            "name": name,
            "country": country,
            "history": history,
            "lifestyle": existing.get("lifestyle", {}),
            "assessments": existing.get("assessments", []),
        }
        save_users(users)

        session["username"] = name
        return redirect(url_for("dashboard", username=name))

    return render_template("login.html")


@app.route("/dashboard/<username>")
def dashboard(username):
    users = load_users()
    user = users.get(username)
    if not user:
        return redirect(url_for("login"))

    wellness = compute_wellness(user.get("lifestyle", {}))
    return render_template(
        "dashboard.html",
        user=user,
        username=username,
        wellness=wellness,
        assessments=list(reversed(user.get("assessments", []))),
    )


@app.route("/lifestyle/<username>", methods=["POST"])
def update_lifestyle(username):
    users = load_users()
    if username not in users:
        return jsonify({"ok": False}), 404

    fields = ["sleep_hours", "sleep_quality", "wake_refreshed",
              "exercise_freq", "exercise_type", "diet_type",
              "processed_food", "alcohol", "smoking", "caffeine"]
    lifestyle = {f: request.form.get(f) for f in fields if request.form.get(f)}

    users[username]["lifestyle"] = lifestyle
    save_users(users)
    return jsonify({"ok": True, "wellness": compute_wellness(lifestyle)})


@app.route("/console/<username>")
def console(username):
    users = load_users()
    user = users.get(username)
    if not user:
        return redirect(url_for("login"))
    return render_template(
        "chatbot.html",
        user=user,
        username=username,
        assessments=list(reversed(user.get("assessments", []))),
    )


@app.route("/ask/<username>", methods=["POST"])
def ask(username):
    users = load_users()
    user = users.get(username)
    if not user:
        return jsonify({"error": "Unknown user. Please sign in again."}), 404

    symptoms = (request.form.get("message1") or "").strip()
    vitals = (request.form.get("message2") or "").strip()
    facial = (request.form.get("message3") or "").strip()

    if not symptoms:
        return jsonify({"error": "Describe at least one symptom to run an assessment."}), 400

    wellness = compute_wellness(user.get("lifestyle", {}))
    prompt = build_prompt(user, wellness, symptoms, vitals, facial)
    result = run_triage(prompt)

    country = user.get("country", "india")
    result["emergency_number"] = EMERGENCY_NUMBERS.get(country, "your local emergency number")

    record = {
        "id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": {"symptoms": symptoms, "vitals": vitals, "facial": facial},
        "result": result,
    }
    user.setdefault("assessments", []).append(record)
    save_users(users)

    return jsonify(record)


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


# --------------------------------------------------------------------------- #
#  RUN
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if not API_KEY:
        print("WARNING: GEMINI_API_KEY is not set. The console will load, but "
              "assessments will show a fallback message until you set the key.")
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
