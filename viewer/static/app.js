"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  index: null,
  mode: "category",      // "category" | "paper"
  category: "physics",
  paper: "",             // slug filter ("" = all papers in category mode)
  current: null,         // {paper, number}
  streaming: false,
  eventSource: null,
};

// ---------- bootstrap ----------------------------------------------------

async function init() {
  const res = await fetch("/api/index");
  if (!res.ok) {
    $("nav-status").textContent = (await res.json()).error || "failed to load index";
    return;
  }
  state.index = await res.json();
  renderModeToggle();
  renderCategoryTabs();
  renderPaperSelect();
  renderGrid();
}

function renderModeToggle() {
  $("mode-category").onclick = () => setMode("category");
  $("mode-paper").onclick = () => setMode("paper");
}

function setMode(mode) {
  state.mode = mode;
  $("mode-category").classList.toggle("active", mode === "category");
  $("mode-paper").classList.toggle("active", mode === "paper");
  if (mode === "paper" && !state.paper) state.paper = state.index.papers[0]?.slug || "";
  renderPaperSelect();
  renderGrid();
}

function renderCategoryTabs() {
  const tabs = $("category-tabs");
  tabs.innerHTML = "";
  for (const cat of state.index.categories) {
    const b = document.createElement("button");
    b.textContent = cat;
    b.classList.toggle("active", cat === state.category);
    b.onclick = () => { state.category = cat; renderCategoryTabs(); renderGrid(); };
    tabs.appendChild(b);
  }
}

function renderPaperSelect() {
  const sel = $("paper-select");
  sel.innerHTML = "";
  if (state.mode === "category") {
    sel.appendChild(new Option("All papers", ""));
  }
  for (const p of state.index.papers) {
    sel.appendChild(new Option(p.title, p.slug));
  }
  sel.value = state.paper;
  sel.onchange = () => { state.paper = sel.value; renderGrid(); };
}

function visibleQuestions() {
  return state.index.questions.filter((q) =>
    q.category === state.category && (!state.paper || q.paper === state.paper));
}

function renderGrid() {
  const grid = $("question-grid");
  grid.innerHTML = "";
  const qs = visibleQuestions();
  for (const q of qs) {
    const b = document.createElement("button");
    b.textContent = q.number;
    const paper = state.index.papers.find((p) => p.slug === q.paper);
    b.title = paper ? paper.title : q.paper;
    b.classList.toggle("active",
      state.current && state.current.paper === q.paper && state.current.number === q.number);
    b.onclick = () => selectQuestion(q.paper, q.number);
    grid.appendChild(b);
  }
  $("nav-status").textContent = `${qs.length} questions`;
}

// ---------- question view -------------------------------------------------

async function selectQuestion(paper, number) {
  closeStream();
  state.current = { paper, number };
  renderGrid();

  const res = await fetch(`/api/papers/${paper}/questions/${number}`);
  if (!res.ok) return;
  const q = await res.json();

  $("placeholder").style.display = "none";
  $("question-header").textContent =
    `Q${q.number} - ${q.category} - ${q.paper_title}`;
  const img = $("question-image");
  img.src = `/img/${paper}/${q.image}`;
  img.classList.remove("zoomed");
  $("question-image-wrap").style.display = "block";
  if (q.text) {
    $("question-text").textContent = q.text;
    $("text-source").textContent = `(${q.text_source})`;
    $("question-text-wrap").style.display = "block";
  } else {
    $("question-text-wrap").style.display = "none";
  }
  await loadChat();
}

$("question-image").onclick = () => $("question-image").classList.toggle("zoomed");

// ---------- chat ----------------------------------------------------------

function chatUrl(suffix = "") {
  return `/api/chat/${state.current.paper}/${state.current.number}${suffix}`;
}

async function loadChat() {
  const box = $("chat-messages");
  box.innerHTML = "";
  if (!state.current) return;
  const res = await fetch(chatUrl());
  const chat = await res.json();
  for (const m of chat.messages) addMsg(m.role, m.content, m);
  if (chat.streaming) attachStream(chat.streaming, addMsg("assistant", "", {}));
  box.scrollTop = box.scrollHeight;
}

function renderInto(el, text) {
  el.innerHTML = renderMarkdown(text);
  if (window.renderMathInElement) {
    try {
      renderMathInElement(el, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
          { left: "$", right: "$", display: false },
        ],
        throwOnError: false,
      });
    } catch (_) { /* partial latex while streaming - ignore */ }
  }
}

function addMsg(role, content, extra) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (role === "assistant") renderInto(div, content);
  else div.innerHTML = escapeHtml(content);
  if (extra && extra.stopped) div.innerHTML += '<div class="msg status">(stopped)</div>';
  $("chat-messages").appendChild(div);
  $("chat-messages").scrollTop = $("chat-messages").scrollHeight;
  return div;
}

function setStreaming(on) {
  state.streaming = on;
  $("chat-send").disabled = on;
  $("chat-stop").disabled = !on;
}

$("chat-form").onsubmit = async (e) => {
  e.preventDefault();
  if (!state.current || state.streaming) return;
  const input = $("chat-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  addMsg("user", message);
  const res = await fetch(chatUrl("/message"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  const data = await res.json();
  if (!res.ok) { addMsg("error", data.error || "request failed"); return; }
  attachStream(data.stream_id, addMsg("assistant", "", {}));
};

$("chat-input").onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("chat-form").requestSubmit(); }
};

function attachStream(streamId, bubble) {
  setStreaming(true);
  bubble.classList.add("cursor");
  let text = "";
  let statusLine = null;
  const es = new EventSource(`/api/chat/stream/${streamId}`);
  state.eventSource = es;

  const clearStatus = () => { if (statusLine) { statusLine.remove(); statusLine = null; } };

  es.addEventListener("delta", (e) => {
    clearStatus();
    text += JSON.parse(e.data).text;
    renderInto(bubble, text);
    $("chat-messages").scrollTop = $("chat-messages").scrollHeight;
  });
  es.addEventListener("status", (e) => {
    clearStatus();
    statusLine = addMsg("status", JSON.parse(e.data).text);
  });
  es.addEventListener("done", (e) => {
    clearStatus();
    const d = JSON.parse(e.data);
    renderInto(bubble, d.full_text || text);
    if (d.stopped) bubble.innerHTML += '<div class="msg status">(stopped)</div>';
    finishStream(es, bubble);
  });
  es.addEventListener("error", (e) => {
    clearStatus();
    if (e.data) addMsg("error", JSON.parse(e.data).message || "assistant error");
    finishStream(es, bubble);
  });
  es.onerror = () => finishStream(es, bubble);   // connection drop
}

function finishStream(es, bubble) {
  bubble.classList.remove("cursor");
  if (bubble.textContent.trim() === "") bubble.remove();
  es.close();
  if (state.eventSource === es) state.eventSource = null;
  setStreaming(false);
}

function closeStream() {
  if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
  setStreaming(false);
}

$("chat-stop").onclick = () => { if (state.current) fetch(chatUrl("/stop"), { method: "POST" }); };

$("chat-reset").onclick = async () => {
  if (!state.current) return;
  if (!confirm("Delete this question's chat history?")) return;
  closeStream();
  await fetch(chatUrl(), { method: "DELETE" });
  loadChat();
};

// ---------- tiny markdown -------------------------------------------------

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderMarkdown(s) {
  let html = escapeHtml(s);
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, l, code) => `<pre><code>${code}</code></pre>`);
  html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/^### (.+)$/gm, "<strong>$1</strong>");
  html = html.replace(/^## (.+)$/gm, "<strong>$1</strong>");
  html = html.replace(/^- (.+)$/gm, "&bull; $1");
  return html.replace(/\n/g, "<br>");
}

init();
