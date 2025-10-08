from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# ✅ Lightweight model (small enough for free Render instance)
MODEL_NAME = "distilgpt2"

# Load model and tokenizer (small size = ~300 MB RAM)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

@app.route("/")
def home():
    return jsonify({"message": "🚀 Code Suggestion API is running!"})

@app.route("/suggest", methods=["POST"])
def suggest():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Generate code suggestion
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_k=50
        )
        suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"prompt": prompt, "suggestion": suggestion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
