from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# 🔥 Load a lightweight pretrained model (good for deployment)
MODEL_NAME = "distilgpt2"  # small & fast
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

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

        # 🔧 Generate code suggestion
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,  # how many tokens to generate
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({
            "prompt": prompt,
            "suggestion": suggestion
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # ✅ Important for Render: Bind to 0.0.0.0 and use PORT env variable
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
