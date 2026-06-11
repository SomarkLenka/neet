"use strict";

const $ = (id) => document.getElementById(id);

const ADMIN = new URLSearchParams(location.search).get("admin") === "1";

const state = {
  index: null,
  bubbles: [],           // bubble tree (client view)
  baked: {},             // node_id -> {status, answer, sources} for current question
  mode: "category",      // "category" | "paper"
  category: "physics",
  paper: "",             // slug filter ("" = all papers in category mode)
  current: null,         // {paper, number}
  streaming: false,
  eventSource: null,
};

// ---------- bootstrap ----------------------------------------------------

async function init() {
  if (ADMIN) document.body.classList.add("admin");
  const res = await fetch("/api/index");
  if (!res.ok) {
    $("nav-status").textContent = (await res.json()).error || "failed to load index";
    return;
  }
  state.index = await res.json();
  try {
    const b = await (await fetch("/api/bubbles")).json();
    state.bubbles = b.bubbles || [];
  } catch (_) { state.bubbles = []; }
  if (ADMIN) $("chat-form").hidden = false;
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
  // baked answers for this question (drives the bubbles)
  try {
    const doc = await (await fetch(bakedUrl())).json();
    state.baked = doc.nodes || {};
  } catch (_) { state.baked = {}; }

  const res = await fetch(chatUrl());
  const chat = await res.json();
  for (const m of chat.messages) addMsg(m.role, m.content, m);
  if (chat.streaming) attachStream(chat.streaming, addMsg("assistant", "", {}));

  // resume the branch exactly as a click would leave it: a node with follow-ups
  // shows them plus "start over"; a leaf node returns to the top-level bubbles.
  const lastNode = [...chat.messages].reverse().find((m) => m.node_id)?.node_id;
  const next = lastNode ? followupsOf(lastNode) : [];
  renderBubbles(next.length ? next : state.bubbles, { showBack: next.length > 0 });

  // a generation started elsewhere is still running for this question: show a
  // pending state and poll until it lands, then re-render with the answer.
  if (chat.generating) {
    const p = document.createElement("div");
    p.className = "msg assistant pending";
    p.textContent = chat.generating.kind === "rag"
      ? "Searching the textbooks..." : "Finding the answer...";
    box.appendChild(p);
    waitForGeneration({ paper: state.current.paper, number: state.current.number });
  }
  box.scrollTop = box.scrollHeight;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitForGeneration(origin) {
  while (isCurrent(origin)) {
    await sleep(3000);
    if (!isCurrent(origin)) return;
    let chat;
    try {
      chat = await (await fetch(`/api/chat/${origin.paper}/${origin.number}`)).json();
    } catch (_) { return; }
    if (!chat.generating) {
      if (isCurrent(origin)) loadChat();   // re-render now that the answer is saved
      return;
    }
  }
}

// ---------- support bubbles ----------------------------------------------

function bakedUrl() {
  return `/api/baked/${state.current.paper}/${state.current.number}`;
}

function findNode(nodeId, tree = state.bubbles) {
  for (const n of tree) {
    if (n.id === nodeId) return n;
    const hit = findNode(nodeId, n.followups || []);
    if (hit) return hit;
  }
  return null;
}

function followupsOf(nodeId) {
  return findNode(nodeId)?.followups || [];
}

function renderBubbles(nodes, { showBack = false } = {}) {
  const bar = $("bubble-bar");
  bar.innerHTML = "";
  if (!state.current || !nodes) return;
  if (showBack) {
    const back = document.createElement("button");
    back.className = "bubble-back";
    back.textContent = "← start over";
    back.onclick = () => renderBubbles(state.bubbles);
    bar.appendChild(back);
  }
  for (const node of nodes) {
    const b = document.createElement("button");
    b.className = "bubble-btn" + (showBack ? " followup" : "");
    b.textContent = node.label;
    b.onclick = () => clickBubble(node);
    bar.appendChild(b);
  }
}

function isCurrent(q) {
  return state.current && q && state.current.paper === q.paper && state.current.number === q.number;
}

async function clickBubble(node) {
  const origin = state.current;                       // question this click belongs to
  const url = `/api/baked/${origin.paper}/${origin.number}`;
  addMsg("user", node.label);
  let pending = null;
  if (node.on_demand) {
    pending = document.createElement("div");
    pending.className = "msg assistant pending";
    pending.textContent = node.kind === "rag"
      ? "Searching the textbooks..." : "Finding the answer...";
    $("chat-messages").appendChild(pending);
    $("chat-messages").scrollTop = $("chat-messages").scrollHeight;
  }
  let data;
  try {
    const res = await fetch(`${url}/click`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: node.id, label: node.label }),
    });
    data = await res.json();
  } catch (e) {
    data = { error: e.message };
  }
  // navigated away while this generated: it's cached + saved to origin's history,
  // so don't draw into whatever question is now on screen.
  if (!isCurrent(origin)) return;
  if (pending) pending.remove();
  if (data.answer) {
    renderAnswer(node, data.answer, data.sources);
  } else if (data.error) {
    addMsg("error", data.error);
  } else {
    renderEmpty(node, data.status);
  }
  const next = node.followups || [];
  renderBubbles(next.length ? next : state.bubbles, { showBack: next.length > 0 });
}

function renderAnswer(node, answer, sources) {
  const div = addMsg("assistant", answer);
  if (sources && sources.length) {
    const s = document.createElement("div");
    s.className = "bubble-sources";
    s.innerHTML = "Sources: " + sources.map((x) =>
      escapeHtml(typeof x === "string" ? x : (x.title || x.page || JSON.stringify(x)))).join(" · ");
    div.appendChild(s);
  }
}

function renderEmpty(node, status) {
  const div = document.createElement("div");
  div.className = "msg assistant pending";
  div.textContent = "Not available yet for this question.";
  $("chat-messages").appendChild(div);
  if (ADMIN) {
    const g = document.createElement("div");
    g.className = "msg gen-btn";
    const btn = document.createElement("button");
    btn.textContent = "⚡ Generate now (admin)";
    btn.onclick = () => generateNode(node, div, btn);
    g.appendChild(btn);
    $("chat-messages").appendChild(g);
  }
  $("chat-messages").scrollTop = $("chat-messages").scrollHeight;
}

async function generateNode(node, pendingDiv, btn) {
  btn.disabled = true;
  btn.textContent = "generating...";
  try {
    const res = await fetch(`${bakedUrl()}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: node.id }),
    });
    const data = await res.json();
    if (!res.ok) { btn.textContent = "error: " + (data.error || "failed"); return; }
    state.baked[node.id] = data;
    pendingDiv.parentElement && btn.parentElement.remove();
    pendingDiv.className = "msg assistant";
    renderInto(pendingDiv, data.answer);
  } catch (e) {
    btn.textContent = "error: " + e.message;
  }
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
  if (!confirm("Clear this question's conversation?")) return;
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
  // [label](url) -> link opening in a new tab; only http(s) or same-origin /path
  html = html.replace(/\[([^\]\n]+)\]\((\/[^)\s]+|https?:\/\/[^)\s]+)\)/g,
    (_, label, url) => `<a href="${url}" target="_blank" rel="noopener">${label}</a>`);
  html = html.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/^### (.+)$/gm, "<strong>$1</strong>");
  html = html.replace(/^## (.+)$/gm, "<strong>$1</strong>");
  html = html.replace(/^- (.+)$/gm, "&bull; $1");
  return html.replace(/\n/g, "<br>");
}

init();
