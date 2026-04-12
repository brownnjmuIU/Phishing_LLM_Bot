const path = require("path");
const fs = require("fs");
const express = require("express");
const OpenAI = require("openai");

const PORT = Number(process.env.PORT) || 3000;

/** OpenAI-compatible Hugging Face Inference Providers router */
const HF_ROUTER_BASE = "https://router.huggingface.co/v1";

/**
 * Default: strong open instruct model (override with HF_MODEL).
 * Browse: https://huggingface.co/models?inference_provider=all
 */
const DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct";

function getHFToken() {
  return (
    process.env.HF_TOKEN ||
    process.env.HUGGING_FACE_HUB_TOKEN ||
    ""
  ).trim();
}

function loadSystemPrompt() {
  const fromEnv = process.env.AGENT_SYSTEM_PROMPT;
  if (fromEnv && fromEnv.trim()) {
    return fromEnv.trim();
  }
  const filePath =
    process.env.AGENT_INSTRUCTIONS_FILE ||
    path.join(__dirname, "config", "agent_instructions.txt");
  try {
    return fs.readFileSync(filePath, "utf8").trim();
  } catch {
    return "You are a helpful assistant.";
  }
}

const SYSTEM_PROMPT = loadSystemPrompt();

const MODEL = process.env.HF_MODEL || DEFAULT_HF_MODEL;

const app = express();
app.use(express.json({ limit: "512kb" }));
app.use(express.static(path.join(__dirname, "public")));

function getClient() {
  const token = getHFToken();
  if (!token) return null;
  return new OpenAI({
    apiKey: token,
    baseURL: HF_ROUTER_BASE,
  });
}

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    provider: "huggingface",
    model: MODEL,
    hasToken: Boolean(getHFToken()),
  });
});

app.post("/api/chat", async (req, res) => {
  const { messages } = req.body || {};
  if (!Array.isArray(messages)) {
    res.status(400).json({ error: "Expected body: { messages: [{ role, content }] }" });
    return;
  }

  const sanitized = messages
    .filter(
      (m) =>
        m &&
        (m.role === "user" || m.role === "assistant") &&
        typeof m.content === "string"
    )
    .slice(-40)
    .map((m) => ({ role: m.role, content: m.content }));

  const client = getClient();
  if (!client) {
    res.status(503).json({
      error:
        "Server is missing HF_TOKEN. Create a free token at huggingface.co/settings/tokens (Inference permissions) and set HF_TOKEN on Render.",
    });
    return;
  }

  try {
    const completion = await client.chat.completions.create({
      model: MODEL,
      messages: [{ role: "system", content: SYSTEM_PROMPT }, ...sanitized],
      temperature: Number(process.env.HF_TEMPERATURE) || 0.7,
      max_tokens: Number(process.env.HF_MAX_TOKENS) || 1024,
    });

    const text = completion.choices[0]?.message?.content ?? "";
    res.json({ reply: text });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Unknown error";
    console.error(err);
    res.status(502).json({ error: msg });
  }
});

app.listen(PORT, () => {
  console.log(`Listening on ${PORT}`);
});
