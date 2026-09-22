// BizInsights AI - Mobile Client Application
document.addEventListener("DOMContentLoaded", () => {
  const chatContainer = document.getElementById("chat-container");
  const messagesList = document.getElementById("messages-list");
  const welcomeCard = document.getElementById("welcome-card");
  const chatForm = document.getElementById("chat-form");
  const promptInput = document.getElementById("prompt-input");
  const sendBtn = document.getElementById("send-btn");
  const btnClear = document.getElementById("btn-clear");
  const btnQr = document.getElementById("btn-qr");
  const qrModal = document.getElementById("qr-modal");
  const modalClose = document.getElementById("modal-close");
  const statusText = document.getElementById("status-text");
  const networkUrlText = document.getElementById("network-url-text");
  const btnCopyUrl = document.getElementById("btn-copy-url");

  let messageHistory = [];
  let isGenerating = false;
  let abortController = null;
  let serverInfo = null;

  const CUSTOM_DOMAIN_URL = "https://bizinsight.oselumeseagbonrofo.dev/";

  // Fetch server and network metadata
  async function fetchServerInfo() {
    try {
      // Prioritize current domain if accessed over web/custom domain, else default to custom domain
      const activeUrl = window.location.origin.startsWith("http") && !window.location.origin.includes("127.0.0.1") && !window.location.origin.includes("localhost")
        ? window.location.origin + "/"
        : CUSTOM_DOMAIN_URL;

      networkUrlText.textContent = activeUrl;

      const res = await fetch("/api/info");
      if (res.ok) {
        serverInfo = await res.json();
        statusText.textContent = serverInfo.status === "ready" ? "Online" : "Ready";
      } else {
        statusText.textContent = "Online";
      }
    } catch (e) {
      statusText.textContent = "Online";
    }
  }

  fetchServerInfo();

  // --- Privacy-Preserving Client Analytics Tracker ---
  function getOrCreateVisitorId() {
    let vid = localStorage.getItem("biz_visitor_id");
    if (!vid) {
      vid = "v_" + Math.random().toString(36).substring(2, 10) + Date.now().toString(36);
      localStorage.setItem("biz_visitor_id", vid);
    }
    return vid;
  }

  function getOrCreateSessionId() {
    let sid = sessionStorage.getItem("biz_session_id");
    if (!sid) {
      sid = "s_" + Math.random().toString(36).substring(2, 10) + Date.now().toString(36);
      sessionStorage.setItem("biz_session_id", sid);
    }
    return sid;
  }

  function detectClientOS() {
    const ua = navigator.userAgent || "";
    if (/iPhone|iPod/.test(ua)) return "iOS";
    if (/iPad/.test(ua)) return "iPadOS";
    if (/Android/.test(ua)) return "Android";
    if (/Win/.test(ua)) return "Windows";
    if (/Macintosh|Mac OS X/.test(ua)) return "macOS";
    if (/CrOS/.test(ua)) return "Chrome OS";
    if (/Linux/.test(ua)) return "Linux";
    return "Unknown";
  }

  function detectClientDevice() {
    const ua = navigator.userAgent || "";
    if (/iPad|Tablet/i.test(ua) || (navigator.maxTouchPoints > 1 && window.innerWidth >= 768 && window.innerWidth <= 1024)) {
      return "Tablet";
    }
    if (/Mobile|iPhone|Android/i.test(ua) || window.innerWidth < 768) {
      return "Mobile";
    }
    return "Desktop";
  }

  function detectClientBrowser() {
    const ua = navigator.userAgent || "";
    if (/Edg\//.test(ua)) return "Microsoft Edge";
    if (/OPR\//.test(ua) || /Opera/.test(ua)) return "Opera";
    if (/SamsungBrowser/.test(ua)) return "Samsung Internet";
    if (/Firefox\//.test(ua)) return "Firefox";
    if (/Chrome\//.test(ua)) return "Chrome";
    if (/Safari\//.test(ua)) return "Safari";
    return "Unknown";
  }

  const visitorId = getOrCreateVisitorId();
  const sessionId = getOrCreateSessionId();

  // --- Offline shop knowledge (RAG) ---
  const btnKb = document.getElementById("btn-kb");
  const kbPanel = document.getElementById("kb-panel");
  const kbEnabled = document.getElementById("kb-enabled");
  const kbFile = document.getElementById("kb-file");
  const kbUpload = document.getElementById("kb-upload");
  const kbStatus = document.getElementById("kb-status");
  const kbList = document.getElementById("kb-list");
  const kbCount = document.getElementById("kb-count");

  if (kbEnabled) {
    try {
      const saved = localStorage.getItem("kb_enabled");
      if (saved !== null) kbEnabled.checked = saved === "1";
    } catch (e) {}
    kbEnabled.addEventListener("change", () => {
      try {
        localStorage.setItem("kb_enabled", kbEnabled.checked ? "1" : "0");
      } catch (e) {}
    });
  }

  if (btnKb && kbPanel) {
    btnKb.addEventListener("click", () => {
      kbPanel.open = !kbPanel.open;
    });
  }

  async function refreshKbList() {
    if (!kbList) return;
    try {
      const res = await fetch("/api/knowledge/list");
      if (!res.ok) return;
      const data = await res.json();
      const docs = data.documents || [];
      kbList.innerHTML = "";
      if (kbCount) kbCount.textContent = docs.length ? `${docs.length} doc${docs.length > 1 ? "s" : ""}` : "";
      if (!docs.length) {
        if (kbStatus) kbStatus.textContent = "No docs yet. Upload a refund policy, price list, or warranty note.";
        // First run hint: open the panel once so the feature is discoverable.
        try {
          if (kbPanel && !localStorage.getItem("kb_seen")) {
            kbPanel.open = true;
            localStorage.setItem("kb_seen", "1");
          }
        } catch (e) {}
        return;
      }
      if (kbStatus) kbStatus.textContent = `${docs.length} doc${docs.length > 1 ? "s" : ""} in use for answers.`;
      docs.forEach((d) => {
        const li = document.createElement("li");
        const label = document.createElement("span");
        label.textContent = `${d.filename} (${d.num_chunks} chunk${d.num_chunks > 1 ? "s" : ""})`;
        const del = document.createElement("button");
        del.className = "kb-del";
        del.textContent = "×";
        del.title = `Remove ${d.filename}`;
        del.setAttribute("aria-label", `Remove ${d.filename}`);
        del.onclick = async () => {
          if (!confirm(`Remove ${d.filename} from shop knowledge?`)) return;
          try {
            const res = await fetch(`/api/knowledge/${d.doc_id}`, { method: "DELETE" });
            if (!res.ok) {
              if (kbStatus) kbStatus.textContent = `Remove failed: ${res.status}. Try again.`;
              return;
            }
          } catch (e) {
            if (kbStatus) kbStatus.textContent = "Remove failed: network error.";
            return;
          }
          refreshKbList();
        };
        li.appendChild(label);
        li.appendChild(del);
        kbList.appendChild(li);
      });
    } catch (e) {}
  }

  if (kbUpload) {
    kbUpload.addEventListener("click", async () => {
      const f = kbFile && kbFile.files && kbFile.files[0];
      if (!f) {
        if (kbStatus) kbStatus.textContent = "Pick a file first (.txt, .md, .csv, .pdf, .docx, max 10 MB).";
        return;
      }
      kbUpload.disabled = true;
      if (kbStatus) kbStatus.textContent = `Uploading ${f.name}...`;
      try {
        const form = new FormData();
        form.append("file", f);
        const res = await fetch("/api/knowledge/upload", { method: "POST", body: form });
        if (res.ok) {
          const info = await res.json();
          if (kbStatus) kbStatus.textContent = `Added ${info.filename} (${info.num_chunks} chunks).`;
          if (kbFile) kbFile.value = "";
        } else {
          const err = await res.json().catch(() => ({}));
          if (kbStatus) kbStatus.textContent = `Upload failed: ${err.detail || res.status}`;
        }
      } catch (e) {
        if (kbStatus) kbStatus.textContent = "Upload failed: network error.";
      } finally {
        kbUpload.disabled = false;
        refreshKbList();
      }
    });
  }

  refreshKbList();

  function sendAnalyticsBeacon(eventType = "pageview") {
    try {
      const payload = {
        visitor_id: visitorId,
        session_id: sessionId,
        event_type: eventType,
        path: window.location.pathname || "/",
        os: detectClientOS(),
        device_type: detectClientDevice(),
        browser: detectClientBrowser(),
        screen_res: `${window.screen.width}x${window.screen.height}`
      };
      fetch("/api/analytics/collect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        keepalive: true
      }).catch(() => {});
    } catch (e) {}
  }

  // Initial pageview tracking
  sendAnalyticsBeacon("pageview");

  // Periodic heartbeat every 45 seconds while tab is active
  setInterval(() => {
    if (document.visibilityState === "visible") {
      sendAnalyticsBeacon("heartbeat");
    }
  }, 45000);

  // Auto-resize input textarea
  promptInput.addEventListener("input", () => {
    promptInput.style.height = "auto";
    promptInput.style.height = Math.min(promptInput.scrollHeight, 120) + "px";
    sendBtn.disabled = promptInput.value.trim().length === 0 && !isGenerating;
  });

  // Handle Enter key (desktop vs mobile)
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && window.innerWidth > 640) {
      e.preventDefault();
      if (!sendBtn.disabled) {
        chatForm.dispatchEvent(new Event("submit"));
      }
    }
  });

  // Handle Preset Click
  document.querySelectorAll(".preset-card").forEach((card) => {
    card.addEventListener("click", () => {
      const prompt = card.getAttribute("data-prompt");
      if (prompt) {
        promptInput.value = prompt;
        promptInput.dispatchEvent(new Event("input"));
        promptInput.focus();
      }
    });
  });

  // QR Modal Handlers
  btnQr.addEventListener("click", () => {
    qrModal.classList.add("active");
  });

  modalClose.addEventListener("click", () => {
    qrModal.classList.remove("active");
  });

  qrModal.addEventListener("click", (e) => {
    if (e.target === qrModal) qrModal.classList.remove("active");
  });

  btnCopyUrl.addEventListener("click", async () => {
    const url = networkUrlText.textContent;
    if (url && url !== "Loading...") {
      await navigator.clipboard.writeText(url);
      btnCopyUrl.textContent = "Copied!";
      setTimeout(() => (btnCopyUrl.textContent = "Copy"), 2000);
    }
  });

  // Clear Chat Handler
  btnClear.addEventListener("click", () => {
    if (confirm("Clear current conversation?")) {
      messageHistory = [];
      messagesList.innerHTML = "";
      welcomeCard.style.display = "block";
    }
  });

  // Form Submit Handler
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (isGenerating) {
      // Allow user to cancel/abort generation
      if (abortController) abortController.abort();
      return;
    }

    const content = promptInput.value.trim();
    if (!content) return;

    // Hide welcome card once first message is sent
    welcomeCard.style.display = "none";

    // Append user message
    appendMessage("user", content);
    messageHistory.push({ role: "user", content: content });

    // Reset input
    promptInput.value = "";
    promptInput.style.height = "auto";
    sendBtn.disabled = true;

    // Start generation (chat event is recorded cleanly by the server with full metadata)
    await streamResponse();
  });

  function appendMessage(role, text) {
    const item = document.createElement("div");
    item.className = `message-item ${role}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = formatMarkdown(text);
    item.appendChild(bubble);

    if (role === "assistant") {
      const meta = document.createElement("div");
      meta.className = "message-meta";

      const copyBtn = document.createElement("button");
      copyBtn.className = "copy-btn";
      copyBtn.innerHTML = `
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg> Copy
      `;
      copyBtn.onclick = async () => {
        await navigator.clipboard.writeText(bubble.innerText);
        copyBtn.innerHTML = "✓ Copied";
        setTimeout(() => {
          copyBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg> Copy
          `;
        }, 2000);
      };

      meta.appendChild(copyBtn);
      item.appendChild(meta);
    }

    messagesList.appendChild(item);
    scrollToBottom();
    return bubble;
  }

  async function streamResponse() {
    isGenerating = true;
    abortController = new AbortController();
    sendBtn.disabled = false;
    sendBtn.innerHTML = `<span style="font-size:12px;font-weight:700">■</span>`;
    sendBtn.title = "Stop generating";

    const assistantBubble = appendMessage("assistant", "");
    assistantBubble.classList.add("streaming-cursor");

    let reasoningText = "";
    let answerText = "";
    let ragSources = [];
    const startTime = performance.now();

    function renderBubble() {
      let html = "";
      if (reasoningText) {
        html += `<details class="thought-box" open><summary>Thinking Process</summary><div class="thought-content">${formatMarkdown(reasoningText)}</div></details>`;
      }
      if (answerText) {
        html += `<div class="answer-content">${formatMarkdown(answerText)}</div>`;
      }
      assistantBubble.innerHTML = html;
      scrollToBottom();
    }

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Visitor-Id": visitorId,
          "X-Session-Id": sessionId
        },
        body: JSON.stringify({
          messages: messageHistory,
          temperature: 0.3,
          max_tokens: 512,
          stream: true,
          use_rag: kbEnabled ? kbEnabled.checked : true
        }),
        signal: abortController.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let doneSeen = false;

      while (!doneSeen) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // Keep last incomplete line in buffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith("data: ")) continue;

          const dataStr = trimmed.slice(6);
          // Drain the rest of this chunk after DONE instead of
          // dropping it, then exit the read loop.
          if (dataStr === "[DONE]") {
            doneSeen = true;
            continue;
          }

          try {
            const parsed = JSON.parse(dataStr);
            if (parsed.rag_sources) {
              ragSources = parsed.rag_sources;
              continue;
            }
            if (parsed.error) {
              answerText += `\n*[Error: ${parsed.error}]*`;
              renderBubble();
              break;
            }
            const delta = parsed.choices?.[0]?.delta;
            if (delta) {
              if (delta.reasoning_content) {
                reasoningText += delta.reasoning_content;
                renderBubble();
              }
              if (delta.content) {
                answerText += delta.content;
                renderBubble();
              }
            }
          } catch (e) {
            // Ignore partial/unparseable json chunks
          }
        }
      }

      messageHistory.push({ role: "assistant", content: answerText });

      // Show which shop docs grounded the answer
      if (ragSources.length) {
        const srcParent = assistantBubble.parentElement;
        const src = document.createElement("div");
        src.className = "rag-sources";
        src.textContent = "Sources: " + ragSources.map((s) => `${s.filename} (chunk ${s.chunk_id + 1}/${s.num_chunks})`).join(" · ");
        srcParent.appendChild(src);
      }

      // Add generation time to meta
      const elapsedSec = ((performance.now() - startTime) / 1000).toFixed(1);
      const parent = assistantBubble.parentElement;
      const meta = parent.querySelector(".message-meta");
      if (meta) {
        const timeBadge = document.createElement("span");
        timeBadge.textContent = `${elapsedSec}s`;
        meta.prepend(timeBadge);
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        answerText += `\n\n*Error: Could not connect to inference backend.*`;
        renderBubble();
      }
    } finally {
      assistantBubble.classList.remove("streaming-cursor");
      isGenerating = false;
      abortController = null;
      sendBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
      `;
      sendBtn.title = "Send message";
      sendBtn.disabled = promptInput.value.trim().length === 0;
      scrollToBottom();
    }
  }

  function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  // Lightweight 100% offline markdown parser
  function formatMarkdown(text) {
    if (!text) return "";

    // Escape HTML special characters
    let html = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Code blocks ```code```
    html = html.replace(/```([\s\S]*?)```/g, (match, p1) => {
      return `<pre><code>${p1.trim()}</code></pre>`;
    });

    // Inline code `code`
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold **text**
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Italic *text*
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");

    // Headings ###
    html = html.replace(/^### (.*$)/gim, '<h3 style="margin:8px 0 4px;font-size:15px;font-weight:700;">$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2 style="margin:10px 0 6px;font-size:16px;font-weight:700;">$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1 style="margin:12px 0 8px;font-size:18px;font-weight:700;">$1</h1>');

    return html;
  }
});
