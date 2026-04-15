import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI

PORT = int(os.getenv("PORT", "3000"))
HF_ROUTER_BASE = "https://router.huggingface.co/v1"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"


def get_hf_token() -> str:
    return (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()


def load_system_prompt() -> str:
    from_env = (os.getenv("AGENT_SYSTEM_PROMPT") or "").strip()
    if from_env:
        return from_env

    file_path = os.getenv("AGENT_INSTRUCTIONS_FILE")
    instructions_path = Path(file_path) if file_path else (BASE_DIR / "config" / "agent_instructions.txt")
    try:
        return instructions_path.read_text(encoding="utf-8").strip()
    except Exception:
        return "You are a helpful assistant."


SYSTEM_PROMPT = load_system_prompt()
MODEL = os.getenv("HF_MODEL", DEFAULT_HF_MODEL)

app = Flask(__name__, static_folder=str(PUBLIC_DIR), static_url_path="")


def get_client():
    token = get_hf_token()
    if not token:
        return None
    return OpenAI(api_key=token, base_url=HF_ROUTER_BASE)


@app.get("/")
def index():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "provider": "huggingface",
            "model": MODEL,
            "hasToken": bool(get_hf_token()),
        }
    )


@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True) or {}
    messages = body.get("messages")
    if not isinstance(messages, list):
        return jsonify({"error": "Expected body: { messages: [{ role, content }] }"}), 400

    sanitized = []
    for message in messages:
        if (
            isinstance(message, dict)
            and message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
        ):
            sanitized.append({"role": message["role"], "content": message["content"]})
    sanitized = sanitized[-40:]

    client = get_client()
    if client is None:
        return (
            jsonify(
                {
                    "error": "Server is missing HF_TOKEN. Create a free token at huggingface.co/settings/tokens (Inference permissions) and set HF_TOKEN on Render."
                }
            ),
            503,
        )

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *sanitized],
            temperature=float(os.getenv("HF_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("HF_MAX_TOKENS", "1024")),
        )
        text = (completion.choices[0].message.content if completion.choices else "") or ""
        return jsonify({"reply": text})
    except Exception as err:
        return jsonify({"error": str(err) or "Unknown error"}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
