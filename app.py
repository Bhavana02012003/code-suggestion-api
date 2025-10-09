from flask import Flask, request, jsonify
from flask_cors import CORS

# ⚡ use only lightweight model
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# ✅ Load a tiny model so Render free tier can handle it
try:
    print("🚀 Loading model...")
    code_suggester = pipeline(
        "text-generation",
        model="sshleifer/tiny-gpt2"  # ✅ fits in free Render memory
    )
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    code_suggester = None

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
        print(f"📝 Received prompt: {prompt}")

        # Keep generation short to avoid Render timeout
        result = code_suggester(prompt, max_length=50, num_return_sequences=1)
        suggestion = result[0]["generated_text"]

        print(f"✅ Generated suggestion: {suggestion}")
        return jsonify({"prompt": prompt, "suggestion": suggestion})
    except Exception as e:
        print("❌ Error in /suggest:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
