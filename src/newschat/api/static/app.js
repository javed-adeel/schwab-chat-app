"use strict";

const SUGGESTIONS = [
  "What price target changes were reported for Intel?",
  "Compare Nvidia and Intel",
  "What did Jim Cramer say about Netflix?",
  "What is going on with quantum computing stocks?",
];
const MAX_HISTORY = 8;

const log = document.getElementById("log");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");
const suggestionsEl = document.getElementById("suggestions");

let history = []; // [{role, content}]
let busy = false;
let counter = 0;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Render answer text, turning [1] / [2, 3] markers into links to the source list.
function renderAnswer(container, text, msgId) {
  container.replaceChildren();
  const pattern = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    container.append(text.slice(last, match.index));
    match[1].split(",").forEach((raw) => {
      const n = raw.trim();
      const link = el("a", "cite", `[${n}]`);
      link.href = `#src-${msgId}-${n}`;
      container.append(link);
    });
    last = match.index + match[0].length;
  }
  container.append(text.slice(last));
}

function renderSources(box, sources, citedNumbers, msgId) {
  box.replaceChildren();
  if (!sources.length) return;
  box.append(el("h2", "", "Sources"));
  const list = el("ol");
  sources.forEach((s) => {
    const item = el("li", citedNumbers.has(s.number) ? "cited" : "");
    item.id = `src-${msgId}-${s.number}`;
    item.value = s.number;
    const link = el("a", "", s.title);
    link.href = s.link;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    item.append(link);
    if (s.tickers.length) item.append(el("span", "badge", s.tickers.join(" · ")));
    if (s.headline_only) item.append(el("span", "badge", "headline only"));
    list.append(item);
  });
  box.append(list);
}

function renderHow(box, plan, retrieval) {
  box.replaceChildren();
  const summary = el("summary", "", "How I read your question");
  const parts = [
    `type: ${plan.intent}`,
    `companies: ${plan.tickers.length ? plan.tickers.join(", ") : "none"}`,
    plan.is_follow_up ? "follow-up: yes" : null,
    `search: “${plan.standalone_query}”`,
    `coverage: ${retrieval.coverage}`,
    retrieval.uncovered_terms.length ? `not in articles: ${retrieval.uncovered_terms.join(", ")}` : null,
  ].filter(Boolean);
  box.append(summary, el("div", "", parts.join(" · ")));
}

function addMessage(role, text) {
  const wrapper = el("div", `msg ${role}`);
  if (text !== undefined) wrapper.append(el("div", "answer", text));
  log.append(wrapper);
  wrapper.scrollIntoView({ block: "end" });
  return wrapper;
}

function parseSSE(buffer) {
  const events = [];
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop();
  for (const block of blocks) {
    let event = "message";
    const data = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7);
      else if (line.startsWith("data: ")) data.push(line.slice(6));
    }
    if (data.length) events.push({ event, data: JSON.parse(data.join("\n")) });
  }
  return { events, rest };
}

async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true;
  sendBtn.disabled = true;
  suggestionsEl.hidden = true;
  addMessage("user", question);
  input.value = "";
  autosize();

  const msgId = ++counter;
  const bubble = addMessage("assistant");
  const answerEl = el("div", "answer", "…");
  const sourcesEl = el("div", "sources");
  const howEl = el("details", "how");
  bubble.append(answerEl, sourcesEl, howEl);

  let text = "";
  let sources = [];
  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, history: history.slice(-MAX_HISTORY) }),
    });
    if (!response.ok) {
      const detail = response.status === 422 ? "That message couldn't be processed." : "Something went wrong.";
      throw new Error(detail);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parsed = parseSSE(buffer);
      buffer = parsed.rest;
      for (const { event, data } of parsed.events) {
        if (event === "meta") {
          sources = data.sources;
          renderSources(sourcesEl, sources, new Set(), msgId);
          renderHow(howEl, data.plan, data.retrieval);
        } else if (event === "delta") {
          text += data.text;
          renderAnswer(answerEl, text, msgId);
        } else if (event === "final") {
          text = data.answer;
          renderAnswer(answerEl, text, msgId);
          renderSources(sourcesEl, data.sources, new Set(data.citations.map((c) => c.number)), msgId);
        } else if (event === "error") {
          throw new Error(data.detail);
        }
      }
      bubble.scrollIntoView({ block: "end" });
    }
    history.push({ role: "user", content: question }, { role: "assistant", content: text });
  } catch (err) {
    bubble.className = "msg error";
    bubble.replaceChildren(el("div", "answer", err.message || "Something went wrong."));
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

function autosize() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

function reset() {
  history = [];
  log.replaceChildren();
  suggestionsEl.hidden = false;
  input.focus();
}

SUGGESTIONS.forEach((q) => {
  const b = el("button", "", q);
  b.type = "button";
  b.addEventListener("click", () => ask(q));
  suggestionsEl.append(b);
});
sendBtn.addEventListener("click", () => ask(input.value));
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    ask(input.value);
  }
});
document.getElementById("new-chat").addEventListener("click", reset);

fetch("/api/health")
  .then((r) => r.json())
  .then((h) => {
    const mode = h.answerer === "extractive" ? "extractive mode (no LLM key)" : h.answerer;
    statusEl.textContent = `${h.articles} articles indexed · ${mode}`;
  })
  .catch(() => { statusEl.textContent = "Service unavailable"; });
