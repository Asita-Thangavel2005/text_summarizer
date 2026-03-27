from flask import Flask, render_template, request, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Fixed URL format for new HF router
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sshleifer/distilbart-cnn-12-6"
HF_TOKEN   = os.getenv("HF_TOKEN")
headers    = {
    "Authorization" : f"Bearer {HF_TOKEN}",
    "Content-Type"  : "application/json"
}

def summarize_text(text):
    response = requests.post(
        HF_API_URL,
        headers = headers,
        json    = {
            "inputs"    : text,
            "parameters": {
                "max_length": 130,
                "min_length": 30
            }
        }
    )
    print("STATUS CODE :", response.status_code)
    print("RESPONSE    :", response.text)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"API Error {response.status_code}: {response.text}"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please enter some text!"})

    if len(text.split()) < 30:
        return jsonify({"error": "Text too short! Please enter at least 30 words."})

    text   = text[:1024]
    result = summarize_text(text)

    print("RESULT:", result)

    if isinstance(result, list):
        summary = result[0]["summary_text"]
    elif isinstance(result, dict) and "error" in result:
        return jsonify({"error": result["error"]})
    else:
        return jsonify({"error": "Unexpected response from API"})

    original_words = len(text.split())
    summary_words  = len(summary.split())
    reduction      = round((1 - summary_words/original_words) * 100)

    return jsonify({
        "summary"        : summary,
        "original_words" : original_words,
        "summary_words"  : summary_words,
        "reduction"      : reduction
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)