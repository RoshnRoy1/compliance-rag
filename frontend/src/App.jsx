import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const EXAMPLES = [
  "What are the penalties for GDPR violations?",
  "What counts as a high-risk AI system under the EU AI Act?",
  "What rights do data subjects have under GDPR?",
  "When must a data breach be reported to the supervisory authority?",
  "Does the AI Act apply to open source models?",
];

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversation, setConversation] = useState([]);
  const [dbStatus, setDbStatus] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [expandedSource, setExpandedSource] = useState(null);
  const [activeTab, setActiveTab] = useState("chat");
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    fetch(API + "/api/health")
      .then((r) => r.json())
      .then((d) => setDbStatus(d))
      .catch(() => setDbStatus(null));

    fetch(API + "/api/documents")
      .then((r) => r.json())
      .then((d) => setDocuments(d.documents || []))
      .catch(() => setDocuments([]));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation, loading]);

  const askQuestion = async (q) => {
    const text = q || question;
    if (!text.trim()) return;
    setLoading(true);
    setActiveTab("chat");
    setConversation((prev) => [...prev, { role: "user", text: text }]);
    setQuestion("");

    try {
      const start = performance.now();
      const res = await fetch(API + "/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, top_k: 3 }),
      });
      const data = await res.json();
      const elapsed = ((performance.now() - start) / 1000).toFixed(1);
      setConversation((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.answer,
          sources: data.sources,
          time: elapsed,
        },
      ]);
    } catch (err) {
      setConversation((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Could not reach the backend. Is the API running?",
          sources: [],
        },
      ]);
    }
    setLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  };

  const clearChat = () => {
    setConversation([]);
    setExpandedSource(null);
  };

  const formatDocName = (filename) => {
    return filename
      .replace("docs/", "")
      .replace(".pdf", "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="logo">
          <div className="logo-icon">C</div>
          <span>ComplianceRAG</span>
        </div>

        <div className="sidebar-section">
          <div className={dbStatus ? "status-badge online" : "status-badge offline"}>
            <span className="status-dot"></span>
            {dbStatus
              ? dbStatus.chunks.toLocaleString() + " chunks indexed"
              : "Backend offline"}
          </div>
        </div>

        <div className="sidebar-tabs">
          <button
            className={activeTab === "chat" ? "tab active" : "tab"}
            onClick={() => setActiveTab("chat")}
          >
            Chat
          </button>
          <button
            className={activeTab === "docs" ? "tab active" : "tab"}
            onClick={() => setActiveTab("docs")}
          >
            Documents
          </button>
        </div>

        {activeTab === "chat" && (
          <div className="sidebar-section">
            <div className="sidebar-label">Try asking</div>
            <div className="example-list">
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  className="example-btn"
                  onClick={() => askQuestion(ex)}
                  disabled={loading}
                >
                  {ex}
                </button>
              ))}
            </div>
            {conversation.length > 0 && (
              <button className="clear-btn" onClick={clearChat}>
                Clear conversation
              </button>
            )}
          </div>
        )}

        {activeTab === "docs" && (
          <div className="sidebar-section">
            <div className="sidebar-label">Upload document</div>
            <label className="upload-area">
              <input
                type="file"
                accept=".pdf,.txt,.md,.docx"
                style={{ display: "none" }}
                onChange={async (e) => {
                  const file = e.target.files[0];
                  if (!file) return;
                  setUploading(true);
                  setUploadMsg("");
                  const formData = new FormData();
                  formData.append("file", file);
                  try {
                    const res = await fetch(API + "/api/upload", {
                      method: "POST",
                      body: formData,
                    });
                    const data = await res.json();
                    if (data.error) {
                      setUploadMsg("Error: " + data.error);
                    } else {
                      setUploadMsg(file.name + " added (" + data.chunks_added + " chunks)");
                      fetch(API + "/api/documents")
                        .then((r) => r.json())
                        .then((d) => setDocuments(d.documents || []));
                      fetch(API + "/api/health")
                        .then((r) => r.json())
                        .then((d) => setDbStatus(d));
                    }
                  } catch (err) {
                    setUploadMsg("Upload failed. Is the backend running?");
                  }
                  setUploading(false);
                  e.target.value = "";
                }}
                disabled={uploading}
              />
              <span>{uploading ? "Processing..." : "Drop or click to upload"}</span>
              <span className="upload-hint">PDF, TXT, DOCX</span>
            </label>
            {uploadMsg && <div className="upload-msg">{uploadMsg}</div>}

            <div className="sidebar-label" style={{ marginTop: "16px" }}>Source documents</div>
            <div className="doc-list">
              {documents.map((doc, i) => (
                <a
                  key={i}
                  className="doc-item"
                  href={API + "/api/documents/" + doc.filename}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <div className="doc-icon">PDF</div>
                  <div className="doc-info">
                    <div className="doc-name">{formatDocName(doc.filename)}</div>
                    <div className="doc-size">{doc.size_mb} MB</div>
                  </div>
                  <div className="doc-open">Open</div>
                </a>
              ))}
            </div>
          </div>
        )}

        <div className="sidebar-footer">
          Built by Roshan Roy
          <br />
          MSc AI, Heriot-Watt University
        </div>
      </aside>

      <main className="chat-area">
        <div className="chat-scroll">
          {conversation.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">C</div>
              <h1>Compliance RAG Assistant</h1>
              <p>
                Ask questions about GDPR, EU AI Act, and UK Data Protection.
                Every answer is grounded in the source documents with citations
                you can verify.
              </p>
              <div className="empty-examples">
                {EXAMPLES.slice(0, 3).map((ex, i) => (
                  <button
                    key={i}
                    className="empty-example-btn"
                    onClick={() => askQuestion(ex)}
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          )}

          {conversation.map((msg, i) => (
            <div key={i} className={"msg " + msg.role}>
              {msg.role === "user" ? (
                <div className="msg-bubble user-bubble">{msg.text}</div>
              ) : (
                <div className="msg-bubble assistant-bubble">
                  <ReactMarkdown>{msg.text}</ReactMarkdown>

                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-section">
                      <div className="sources-header">
                        <span>Sources</span>
                        {msg.time && (
                          <span className="response-time">{msg.time}s</span>
                        )}
                      </div>
                      <div className="sources-bar">
                        {msg.sources.map((s, j) => {
                          const key = i + "-" + j;
                          const isOpen = expandedSource === key;
                          const relevance = s.rerank_score !== undefined
                            ? s.rerank_score
                            : (1 - s.distance);
                          const matchPct = Math.max(
                            1,
                            Math.min(99, Math.round(sigmoid((relevance + 4) / 2) * 100))
                          );
                          return (
                            <div key={j} className="source-wrapper">
                              <button
                                className={isOpen ? "source-chip open" : "source-chip"}
                                onClick={() =>
                                  setExpandedSource(isOpen ? null : key)
                                }
                              >
                                <span className="source-name">
                                  {formatDocName(s.source)}
                                </span>
                                <span className="source-meta">p.{s.page}</span>
                                <span className="source-score">
                                  {matchPct}% match
                                </span>
                                <span className="source-toggle">
                                  {isOpen ? "▾" : "▸"}
                                </span>
                              </button>
                              {isOpen && (
                                <div className="source-expanded">
                                  <div className="source-text">{s.text}</div>
                                  <a
                                    className="source-pdf-link"
                                    href={
                                      API +
                                      "/api/documents/" +
                                      s.source.replace("docs/", "")
                                    }
                                    target="_blank"
                                    rel="noopener noreferrer"
                                  >
                                    Open full PDF
                                  </a>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="msg assistant">
              <div className="msg-bubble assistant-bubble loading-bubble">
                <div className="typing">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                Searching documents...
              </div>
            </div>
          )}

          <div ref={bottomRef}></div>
        </div>

        <div className="input-bar">
          <div className="input-wrap">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a compliance question..."
              rows={1}
              disabled={loading}
            ></textarea>
            <button
              className="send-btn"
              onClick={() => askQuestion()}
              disabled={loading || !question.trim()}
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </button>
          </div>
          <div className="input-hint">
            Press Enter to send. Answers are grounded in indexed documents only.
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;