import { useState } from "react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const askQuestion = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setAnswer(null);
    setSources([]);

    try {
      const res = await fetch("http://localhost:8000/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 3 }),
      });
      const data = await res.json();
      setAnswer(data.answer);
      setSources(data.sources);
      setHistory((prev) => [
        { question: data.question, answer: data.answer },
        ...prev,
      ]);
    } catch (err) {
      setAnswer("Error connecting to the API. Is the backend running?");
    }
    setLoading(false);
    setQuestion("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Compliance RAG Assistant</h1>
        <p>Ask questions about GDPR, EU AI Act, and UK Data Protection Act</p>
      </header>

      <main className="main">
        {answer && (
          <div className="answer-card">
            <div className="answer-label">Answer</div>
            <div className="answer-text">{answer}</div>
            {sources.length > 0 && (
              <div className="sources">
                <div className="sources-label">Sources</div>
                {sources.map((s, i) => (
                  <div key={i} className="source-item">
                    <span className="source-file">
                      {s.source.replace("docs/", "")}
                    </span>
                    <span className="source-page">Page {s.page}</span>
                    <span className="source-dist">
                      Distance: {s.distance}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {loading && (
          <div className="loading">
            Searching documents and generating answer...
          </div>
        )}

        <div className="input-area">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a compliance question..."
            rows={2}
          />
          <button onClick={askQuestion} disabled={loading || !question.trim()}>
            {loading ? "..." : "Ask"}
          </button>
        </div>

        {history.length > 1 && (
          <div className="history">
            <div className="history-label">Previous questions</div>
            {history.slice(1).map((h, i) => (
              <div key={i} className="history-item">
                <div className="history-q">{h.question}</div>
                <div className="history-a">{h.answer.slice(0, 150)}...</div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;