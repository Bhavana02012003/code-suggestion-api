import os
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from threading import Lock

app = Flask(__name__)

# Keep HF caches small & local
os.environ.setdefault("HF_HOME", "/opt/render/project/.cache/huggingface")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Tiny model (fits free tier)
MODEL_ID = "sshleifer/tiny-gpt2"

_model = None
_tokenizer = None
_model_lock = Lock()

def load_model_once():
    global _model, _tokenizer
    with _model_lock:
        if _model is None or _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
            _model.eval()

@app.get("/")
def root():
    return jsonify({"message": "🚀 Code Suggestion API is running!"})

@app.post("/suggest")
def suggest():
    """
    Body JSON:
    {
      "prompt": "def fibonacci(",
      "max_new_tokens": 16,   # optional, default 24
      "temperature": 0.7      # optional, default 0.7
    }
    Returns short autocomplete.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "Missing 'prompt'."}), 400

        max_new_tokens = int(data.get("max_new_tokens", 24))
        temperature = float(data.get("temperature", 0.7))

        # Guardrails for free tier & tiny model
        max_new_tokens = max(1, min(max_new_tokens, 48))
        temperature = max(0.1, min(temperature, 1.5))

        load_model_once()
        input_ids = _tokenizer.encode(prompt, return_tensors="pt")
        # Very small generation to keep memory & latency down
        with torch.no_grad():
            out_ids = _model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=_tokenizer.eos_token_id
            )
        full_text = _tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # Return ONLY the new bit (autocomplete)
        suggestion = full_text[len(prompt):]

        # Safety: trim weird long tails
        suggestion = suggestion[:256]

        return jsonify({
            "prompt": prompt,
            "suggestion": suggestion
        })
    except Exception as e:
        return jsonify({"error": f"failed: {type(e).__name__}: {str(e)}"}), 500

if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
