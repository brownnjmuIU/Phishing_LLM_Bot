import os
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

# Fallback if the client sends nothing (not recommended for production).
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "DEFAULT_SYSTEM_PROMPT",
    "Follow the operator instructions you are given. Stay in character. Keep replies short (2–5 sentences).",
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    system_prompt: Optional[str] = Field(
        default=None,
        description="Full system / operator instructions for the model (role, goals, style).",
    )


@app.on_event("startup")
async def startup_event():
    global model, generator
    print("Loading Phi-3.5-mini-instruct — first run may take a while...")

    try:
        model_name = "microsoft/Phi-3.5-mini-instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            load_in_4bit=True,
            quantization_config={"load_in_4bit": True},
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.float16,
            max_new_tokens=400,
            temperature=0.75,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        print("Model ready.")
    except Exception as e:
        print(f"Model load failed: {e}")


@app.get("/api/health")
async def health():
    return {"ok": True, "model_ready": generator is not None}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "default_system_prompt": DEFAULT_SYSTEM_PROMPT},
    )


def _build_prompt(system_prompt: str, messages: List[Message]) -> str:
    block = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    parts = [block, ""]
    for msg in messages:
        if msg.role == "user":
            parts.append(f"User: {msg.content}")
        else:
            parts.append(f"Assistant: {msg.content}")
    parts.append("Assistant:")
    return "\n".join(parts)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if generator is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is still loading or failed to load."},
        )

    try:
        system = (request.system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
        conversation = _build_prompt(system, request.messages)

        outputs = generator(
            conversation,
            max_new_tokens=400,
            temperature=0.75,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=generator.tokenizer.eos_token_id,
        )

        response_text = outputs[0]["generated_text"]
        if "Assistant:" in response_text:
            ai_message = response_text.split("Assistant:")[-1].strip()
        else:
            ai_message = response_text.strip()

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
