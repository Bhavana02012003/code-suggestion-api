from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# ✅ Tiny model for free-tier (very lightweight)
# This avoids Render memory crash
try:
    code_suggester = pipeline(
        "text-generation",
        model="sshleifer/tiny-gpt2"  # very small model
    )
except Exception as e:
    code_suggester = None
    print("❌ Failed to load model:", e)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🚀 Code Suggestion API is running!"})

@app.route("/suggest", methods=["POST"])
def suggest_code():
    try:
        if code_suggester is None:
            return jsonify({"error": "Model not loaded on server"}), 500

        data = request.get_json(force=True)
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        prompt = data["prompt"]
        result = code_suggester(prompt, max_length=80, num_return_sequences=1)
        suggestion = result[0]["generated_text"]

        return jsonify({"prompt": prompt, "suggestion": suggestion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
