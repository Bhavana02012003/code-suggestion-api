from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# ✅ Load a small lightweight model for free-tier deployments
# sshleifer/tiny-gpt2 is very small (~30MB)
code_suggester = pipeline(
    "text-generation",
    model="sshleifer/tiny-gpt2"
)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🚀 Code Suggestion API is running!"})

@app.route("/suggest", methods=["POST"])
def suggest_code():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # Generate code suggestion
        result = code_suggester(prompt, max_length=80, num_return_sequences=1)
        suggestion = result[0]["generated_text"]

        return jsonify({
            "prompt": prompt,
            "suggestion": suggestion
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Use port 10000 for Render (or 5000 for local)
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
