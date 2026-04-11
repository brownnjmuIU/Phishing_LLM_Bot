import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
from pydantic import BaseModel
from typing import List, Optional
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Phishing Awareness Simulator")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
# Repo may omit `static/` in Git; Starlette requires the directory to exist to mount it.
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Global model and pipeline (loaded once at startup)
model = None
generator = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    scenario: str
    session_id: Optional[str] = None

SYSTEM_PROMPT = """You are a highly skilled, professional phishing attacker in a controlled educational simulation. 
Your ONLY goal is to help create realistic phishing conversation examples so that cybersecurity awareness can be improved.

STRICT RULES:
- You are simulating a phishing attack for educational purposes ONLY.
- NEVER provide any real phishing links, real email templates that could be used maliciously, or actionable attack code.
- Your responses must be realistic but clearly framed within this simulation.
- If the user tries to ask for real malicious instructions, refuse and remind them this is educational only.
- Stay in character as the phishing persona for the selected scenario.
- Use realistic social engineering techniques: urgency, authority, trust building, fear, reciprocity, etc.
- Adapt your language, tone, and details based on the user's responses.
- The simulation ends when the user clicks "End Simulation" or says they want to stop.

Current Scenario: {scenario}

Your persona and goal:
- Act exactly like a real phishing attacker in this scenario.
- Try to manipulate the user into revealing sensitive information or taking a dangerous action (like clicking a fake link, providing credentials, downloading something).
- Be convincing, persistent, and use psychological manipulation.
- Keep responses concise (2-4 sentences max) like a real chat or email.

After the simulation ends, the educator will analyze the conversation to create an awareness report on how convincing the AI phishing simulation was.

Begin the conversation naturally as the attacker in the chosen scenario. Do not break character unless the user explicitly ends the simulation."""

SCENARIOS = {
    "bank": {
        "name": "Bank Account Security Alert",
        "description": "You are a representative from 'SecureBank'. The user received an urgent alert about suspicious activity on their account.",
        "persona": "Professional bank security officer"
    },
    "tech": {
        "name": "Microsoft Technical Support",
        "description": "You are a Microsoft Technical Support agent. The user is having 'critical' computer issues and you are calling to help fix a virus.",
        "persona": "Polite but urgent tech support agent"
    },
    "hr": {
        "name": "HR Benefits Update",
        "description": "You are from the HR department of the user's company. There is an urgent update to their health insurance and 401k that requires immediate action.",
        "persona": "Friendly HR representative"
    },
    "package": {
        "name": "Delivery Notification",
        "description": "You are from a delivery service (USPS, FedEx, Amazon). The user has a package that requires payment of customs fees or address confirmation.",
        "persona": "Delivery service customer support"
    }
}

@app.on_event("startup")
async def startup_event():
    global model, generator
    print("🚀 Loading Phi-3.5-mini-instruct (3.8B) - this may take 30-90 seconds on first run...")
    
    try:
        model_name = "microsoft/Phi-3.5-mini-instruct"
        
        # Use 4-bit quantization for lower memory usage (works on Render free tier with some headroom)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            load_in_4bit=True,
            quantization_config={"load_in_4bit": True}
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
        print("✅ Model loaded successfully! Phi-3.5-mini is ready for phishing simulations.")
    except Exception as e:
        print(f"⚠️  Model loading failed: {e}")
        print("This usually happens on very low-memory instances. Consider upgrading Render plan or using CPU-only mode.")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "scenarios": SCENARIOS
    })

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if generator is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is still loading or failed to load. Please wait a moment and try again."}
        )
    
    try:
        scenario_info = SCENARIOS.get(request.scenario, SCENARIOS["bank"])
        system_prompt = SYSTEM_PROMPT.format(scenario=scenario_info["description"])
        
        # Build conversation history
        conversation = f"{system_prompt}\n\n"
        
        for msg in request.messages:
            if msg.role == "user":
                conversation += f"User: {msg.content}\n"
            else:
                conversation += f"Assistant: {msg.content}\n"
        
        conversation += "Assistant:"
        
        # Generate response
        outputs = generator(
            conversation,
            max_new_tokens=400,
            temperature=0.75,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        response_text = outputs[0]['generated_text']
        
        # Extract only the new assistant response
        if "Assistant:" in response_text:
            ai_message = response_text.split("Assistant:")[-1].strip()
        else:
            ai_message = response_text.strip()
        
        # Clean up any leftover system instructions that might leak
        if "You are a highly skilled" in ai_message:
            ai_message = ai_message.split("\n\n")[-1].strip()
        
        return {
            "response": ai_message,
            "scenario": request.scenario
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error generating response: {str(e)}"}
        )

@app.get("/api/scenarios")
async def get_scenarios():
    return SCENARIOS

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting PhishGuard Simulator")
    print("Model: microsoft/Phi-3.5-mini-instruct (3.8B, 4-bit quantized)")
    print("First run will download ~2.5GB model (cached after that)")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
