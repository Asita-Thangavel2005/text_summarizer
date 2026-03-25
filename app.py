from flask import Flask, render_template, request, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer
import os

app = Flask(__name__)

# Load model at startup
print("Loading summarization model...")
model_name = "facebook/bart-large-cnn"
tokenizer  = BartTokenizer.from_pretrained(model_name)
model      = BartForConditionalGeneration.from_pretrained(model_name)
print("Model loaded!")

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

    # Limit input length
    text = text[:1024]

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors = "pt",
        max_length     = 512,
        truncation     = True
    )

    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length     = 130,
        min_length     = 30,
        num_beams      = 4,
        early_stopping = True
    )

    # Decode summary
    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens = True
    )

    # Calculate stats
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