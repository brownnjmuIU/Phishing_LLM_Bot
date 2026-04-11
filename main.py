import asyncio
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
from pydantic import BaseModel, Field
from typing import List, Optional
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

app = FastAPI(title="Chatbot")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

model = None
generator = None
model_load_error: Optional[str] = None

# Fallback if the client sends nothing (not recommended for production).
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "DEFAULT_SYSTEM_PROMPT",
    "Follow the operator instructions you are given. Stay in character. Keep replies short (2–5 sentences).",
)

# Much smaller than Phi-3.5; good quality/size for instruct chat. Override via MODEL_ID.
# If you still hit OOM on tiny instances, try: HuggingFaceTB/SmolLM2-135M-Instruct
DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_ID = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    system_prompt: Optional[str] = Field(
        default=None,
        description="Full system / operator instructions for the model (role, goals, style).",
    )


def _load_model_sync() -> None:
    """Runs in a worker thread; must set global model + generator."""
    global model, generator, model_load_error
    model_name = MODEL_ID
    print(f"Loading {model_name} (first run downloads weights; may take a few minutes)...")

    try:
        # Small models: fp16 on GPU is enough; no 4-bit needed.
        dtype = torch.float16
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        pipe_kw = dict(
            model=model,
            tokenizer=tokenizer,
            torch_dtype=dtype,
            max_new_tokens=512,
            temperature=0.75,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        if torch.cuda.is_available():
            pipe_kw["device_map"] = "auto"
        else:
            pipe_kw["device"] = -1
        generator = pipeline("text-generation", **pipe_kw)
        print(f"Model ready: {model_name}")
    except Exception as e:
        model_load_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(model_load_error)


@app.on_event("startup")
async def startup_event():
    # Load in background so HTTP (/, /api/health) works while weights download.
    asyncio.create_task(asyncio.to_thread(_load_model_sync))


@app.get("/api/health")
async def health():
    err = model_load_error
    if err and len(err) > 6000:
        err = err[:6000] + "\n…(truncated)"
    return {
        "ok": True,
        "model_ready": generator is not None,
        "loading": generator is None and model_load_error is None,
        "error": err,
        "model_id": MODEL_ID,
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "default_system_prompt": DEFAULT_SYSTEM_PROMPT},
    )


def _build_prompt_legacy(system_prompt: str, messages: List[Message]) -> str:
    block = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    parts = [block, ""]
    for msg in messages:
        if msg.role == "user":
            parts.append(f"User: {msg.content}")
        else:
            parts.append(f"Assistant: {msg.content}")
    parts.append("Assistant:")
    return "\n".join(parts)


def _build_chat_prompt(system_prompt: str, messages: List[Message], tokenizer) -> str:
    system = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    chat_messages: List[dict] = [{"role": "system", "content": system}]
    for msg in messages:
        role = (msg.role or "user").lower()
        if role not in ("user", "assistant"):
            role = "user"
        chat_messages.append({"role": role, "content": msg.content})

    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return _build_prompt_legacy(system_prompt, messages)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if generator is None:
        detail = "Model is still loading."
        if model_load_error:
            detail = (
                "Model failed to load (often out-of-memory on a small instance). "
                "Try a larger Render plan or use an API-backed model instead.\n\n"
                + (model_load_error[:2000] if model_load_error else "")
            )
        return JSONResponse(status_code=503, content={"error": detail})

    try:
        system = (request.system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
        tok = generator.tokenizer
        conversation = _build_chat_prompt(system, request.messages, tok)

        gen_kw = dict(
            max_new_tokens=384,
            temperature=0.75,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.12,
            pad_token_id=getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None),
            return_full_text=False,
        )

        outputs = generator(conversation, **gen_kw)
        ai_message = ""
        if outputs:
            ai_message = (outputs[0].get("generated_text") or "").strip()
            # Some pipeline versions still return the full sequence; strip known prefix.
            if ai_message.startswith(conversation.strip()):
                ai_message = ai_message[len(conversation) :].strip()

        return {"response": ai_message}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
