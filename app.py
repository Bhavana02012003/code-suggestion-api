from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# 🧠 Load the model only once when the server starts (not for every request)
try:
    generator = pipeline("text-generation", model="microsoft/CodeGPT-small-py")
except Exception as e:
    generator = None
    print("⚠️ Model failed to load:", e)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🚀 Code Suggestion API is running!"})

@app.route("/suggest", methods=["POST"])
def suggest():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        if generator is None:
            return jsonify({"error": "Model could not be loaded due to memory or dependency issues."}), 500

        result = generator(prompt, max_length=100, num_return_sequences=1)
        suggestion = result[0]["generated_text"]

        return jsonify({"prompt": prompt, "suggestion": suggestion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
