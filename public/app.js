const chatEl = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");

/** @type {{ role: 'user' | 'assistant', content: string }[]} */
const messages = [];

function addBubble(role, content, extraClass) {
  const div = document.createElement("div");
  div.className = `msg ${role}${extraClass ? ` ${extraClass}` : ""}`;
  div.textContent = content;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function setLoading(on) {
  sendBtn.disabled = on;
  input.disabled = on;
  statusEl.textContent = on ? "Thinking…" : "";
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  input.value = "";
  messages.push({ role: "user", content: text });
  addBubble("user", text);

  setLoading(true);
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = data.error || res.statusText || "Request failed";
      addBubble("assistant", err, "error");
      messages.pop();
      return;
    }
    const reply = data.reply ?? "";
    messages.push({ role: "assistant", content: reply });
    addBubble("assistant", reply || "(Empty response)");
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Network error";
    addBubble("assistant", msg, "error");
    messages.pop();
  } finally {
    setLoading(false);
    input.focus();
  }
});

fetch("/api/health")
  .then((r) => r.json())
  .then((d) => {
    if (!d.hasToken) {
      statusEl.textContent =
        "Server reports HF_TOKEN missing — add a free Hugging Face token on Render.";
    }
  })
  .catch(() => {});

input.focus();
