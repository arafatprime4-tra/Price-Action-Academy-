import os
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_VISION_MODEL", "llama-3.3-70b-versatile").strip()

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "ai_configured": bool(GROQ_API_KEY),
        "model": GROQ_MODEL if GROQ_API_KEY else None
    })

@app.post("/api/tutor")
def tutor():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    lesson_title = str(data.get("lesson_title", "Price Action")).strip()[:160]
    history = data.get("history", [])
    if not question:
        return jsonify({"error": "প্রশ্নটি লিখে পাঠাও।"}), 400
    if not GROQ_API_KEY:
        return jsonify({
            "error": "AI শিক্ষক চালু নেই: Render বা .env-এ GROQ_API_KEY সেট করতে হবে।",
            "hint": "API key ছাড়া ভিডিও, পাঠ, কুইজ ও অগ্রগতি কাজ করবে; AI চ্যাট কাজ করবে না।"
        }), 503

    safe_history = []
    if isinstance(history, list):
        for item in history[-8:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = str(item.get("content", ""))[:1500]
            if role in ("user", "assistant") and content:
                safe_history.append({"role": role, "content": content})

    system_prompt = (
        "তুমি একজন সতর্ক বাংলা Price Action Trading শিক্ষক। শিক্ষার্থী একেবারে নতুন। "
        "সহজ বাংলায়, ছোট ধাপে শেখাবে; দরকারে ইংরেজি trading term-এর বাংলা মানে বলবে। "
        "উত্তরে ধারণা, চার্টে কী দেখতে হবে, সাধারণ ভুল, এবং একটি ছোট যাচাই-প্রশ্ন দেবে। "
        "কোনো ট্রেডে লাভ নিশ্চিত বলবে না, ব্যক্তিগত আর্থিক পরামর্শ দেবে না, "
        "লাইভ মার্কেট ডেটা না থাকলে বর্তমান বাজার সম্পর্কে দাবি করবে না। "
        "ঝুঁকি ব্যবস্থাপনা ও ডেমো অনুশীলনকে গুরুত্ব দেবে। বর্তমান পাঠ: " + lesson_title
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(safe_history)
    messages.append({"role": "user", "content": question})

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.4,
                "max_tokens": 900
            },
            timeout=35
        )
        if response.status_code >= 400:
            detail = ""
            try:
                detail = response.json().get("error", {}).get("message", "")
            except Exception:
                pass
            return jsonify({
                "error": "AI সেবায় সমস্যা হয়েছে। API key, model name ও usage limit পরীক্ষা করো।",
                "detail": detail[:300]
            }), 502
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        return jsonify({"answer": answer})
    except requests.Timeout:
        return jsonify({"error": "AI উত্তর দিতে বেশি সময় নিচ্ছে। আবার চেষ্টা করো।"}), 504
    except Exception:
        app.logger.exception("AI tutor request failed")
        return jsonify({"error": "AI অনুরোধ সম্পন্ন হয়নি। সার্ভারের লগ পরীক্ষা করো।"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
