from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# ✅ Load a lightweight open-source model (you can change it if needed)
MODEL_NAME = "microsoft/CodeGPT-small-py"  # Good balance between size and code quality

print("⏳ Loading model... (first time may take ~1 min)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("✅ Model loaded successfully!")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Code Suggestion API is running!"})

@app.route("/suggest", methods=["POST"])
def suggest_fix():
    """
    Expects JSON:
    {
        "rule": "C0103",
        "problem": "Variable name does not follow snake_case",
        "code": "Var = 10"
    }
    Returns AI-suggested fix & explanation
    """
    try:
        data = request.get_json()
        rule = data.get("rule", "")
        problem = data.get("problem", "")
        code = data.get("code", "")

        if not code:
            return jsonify({"error": "Missing 'code' in request"}), 400

        # 🔥 Create an instruction-style prompt
        prompt = (
            f"You are a senior code reviewer. The following rule was violated: {rule}.\n"
            f"Problem: {problem}\n"
            f"Here is the code snippet:\n\n{code}\n\n"
            "👉 Suggest a corrected version of the code and briefly explain the fix."
        )

        # 🧠 Generate suggestion
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # ✂️ Post-process suggestion to show only the useful part
        if "👉" in generated:
            suggestion = generated.split("👉")[-1].strip()
        else:
            suggestion = generated.strip()

        return jsonify({
            "rule": rule,
            "problem": problem,
            "original_code": code,
            "suggestion": suggestion
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 👇 Important: Render looks for `app:app` entrypoint (so keep this name)
    app.run(host="0.0.0.0", port=5000)
