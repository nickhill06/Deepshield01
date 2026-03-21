import { useState } from "react";
import axios from "axios";

const API_URL = "http://localhost:8000";

export default function App() {
  const [file, setFile]               = useState(null);
  const [dragging, setDragging]       = useState(false);
  const [loading, setLoading]         = useState(false);
  const [result, setResult]           = useState(null);
  const [error, setError]             = useState(null);
  const [history, setHistory]         = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  };

  const analyzeVideo = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await axios.post(`${API_URL}/api/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Something went wrong!");
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/history`);
      setHistory(response.data.predictions);
      setShowHistory(true);
    } catch (err) {
      setError("Could not load history");
    }
  };

  return (
    <div style={{
      minHeight: "100vh",
      width: "100vw",
      background: "linear-gradient(135deg, #0a0a0f 0%, #0f172a 50%, #0a0a0f 100%)",
      color: "#e2e8f0",
      fontFamily: "'Segoe UI', sans-serif",
      boxSizing: "border-box",
      overflowX: "hidden"
    }}>

      {/* TOP NAV */}
      <nav style={{
        width: "100%",
        padding: "20px 60px",
        borderBottom: "1px solid rgba(0,212,255,0.15)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        boxSizing: "border-box",
        background: "rgba(10,10,15,0.8)",
        backdropFilter: "blur(10px)",
        position: "sticky",
        top: 0,
        zIndex: 100
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 28 }}>🛡️</span>
          <span style={{
            fontSize: 24,
            fontWeight: 900,
            background: "linear-gradient(135deg, #00d4ff, #a78bfa)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent"
          }}>DeepShield</span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={loadHistory}
            style={{
              padding: "8px 20px",
              background: "transparent",
              border: "1px solid #334155",
              borderRadius: 8,
              color: "#94a3b8",
              cursor: "pointer",
              fontSize: 13
            }}
          >
            📋 History
          </button>
        </div>
      </nav>

      {/* HERO SECTION */}
      <div style={{
        textAlign: "center",
        padding: "60px 20px 40px",
        background: "radial-gradient(ellipse at top, rgba(0,212,255,0.08) 0%, transparent 60%)"
      }}>
        <div style={{
          display: "inline-block",
          padding: "6px 16px",
          background: "rgba(0,212,255,0.1)",
          border: "1px solid rgba(0,212,255,0.3)",
          borderRadius: 20,
          fontSize: 12,
          color: "#00d4ff",
          letterSpacing: 2,
          marginBottom: 20
        }}>
          POWERED BY VISION TRANSFORMER (ViT)
        </div>
        <h1 style={{
          fontSize: "clamp(32px, 5vw, 64px)",
          fontWeight: 900,
          margin: "0 0 16px",
          background: "linear-gradient(135deg, #ffffff 0%, #00d4ff 50%, #a78bfa 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          lineHeight: 1.1
        }}>
          Detect Deepfake Videos<br />with AI Precision
        </h1>
        <p style={{
          color: "#64748b",
          fontSize: 18,
          maxWidth: 600,
          margin: "0 auto 40px"
        }}>
          Upload any video and our AI will analyze every frame,
          detect manipulation, and show you exactly where the fake regions are.
        </p>

        {/* STATS ROW */}
        <div style={{
          display: "flex",
          justifyContent: "center",
          gap: 40,
          marginBottom: 48,
          flexWrap: "wrap"
        }}>
          {[
            { value: "80%+", label: "Accuracy" },
            { value: "ViT", label: "Model" },
            { value: "Grad-CAM", label: "Explainable AI" },
            { value: "Real-time", label: "Analysis" }
          ].map((stat, i) => (
            <div key={i} style={{ textAlign: "center" }}>
              <div style={{
                fontSize: 24,
                fontWeight: 900,
                color: "#00d4ff"
              }}>{stat.value}</div>
              <div style={{ fontSize: 12, color: "#475569" }}>{stat.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div style={{
        maxWidth: 1200,
        margin: "0 auto",
        padding: "0 40px 60px",
        boxSizing: "border-box"
      }}>

        {/* TWO COLUMN LAYOUT */}
        <div style={{
          display: "grid",
          gridTemplateColumns: result || loading ? "1fr 1fr" : "1fr",
          gap: 32,
          alignItems: "start"
        }}>

          {/* LEFT: UPLOAD */}
          <div>
            {/* DROP ZONE */}
            <div
              onDrop={handleDrop}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              style={{
                border: `2px dashed ${dragging ? "#00d4ff" : "#334155"}`,
                borderRadius: 20,
                padding: "50px 30px",
                textAlign: "center",
                background: dragging
                  ? "rgba(0,212,255,0.05)"
                  : "rgba(15,23,42,0.8)",
                transition: "all 0.3s",
                marginBottom: 16,
                backdropFilter: "blur(10px)"
              }}
            >
              <div style={{ fontSize: 56, marginBottom: 16 }}>🎬</div>
              <p style={{ fontSize: 18, color: "#94a3b8", marginBottom: 8 }}>
                Drag & drop your video here
              </p>
              <p style={{ color: "#475569", fontSize: 13, marginBottom: 24 }}>
                Supports: MP4, AVI, MOV, MKV
              </p>

              <label style={{
                display: "inline-block",
                padding: "12px 32px",
                background: "rgba(0,212,255,0.1)",
                border: "1px solid rgba(0,212,255,0.4)",
                borderRadius: 10,
                color: "#00d4ff",
                cursor: "pointer",
                fontSize: 14,
                fontWeight: 600,
                transition: "all 0.2s"
              }}>
                📁 Browse File
                <input
                  type="file"
                  accept=".mp4,.avi,.mov,.mkv"
                  style={{ display: "none" }}
                  onChange={(e) => setFile(e.target.files[0])}
                />
              </label>

              {file && (
                <div style={{
                  marginTop: 20,
                  padding: "12px 20px",
                  background: "rgba(52,211,153,0.1)",
                  border: "1px solid rgba(52,211,153,0.3)",
                  borderRadius: 10,
                  color: "#34d399",
                  fontSize: 14
                }}>
                  ✅ {file.name}<br />
                  <span style={{ color: "#475569", fontSize: 12 }}>
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
              )}
            </div>

            {/* ANALYZE BUTTON */}
            <button
              onClick={analyzeVideo}
              disabled={!file || loading}
              style={{
                width: "100%",
                padding: "18px",
                background: file && !loading
                  ? "linear-gradient(135deg, #00d4ff, #a78bfa)"
                  : "#1e293b",
                border: "none",
                borderRadius: 12,
                color: file && !loading ? "#000814" : "#475569",
                fontSize: 18,
                fontWeight: 800,
                cursor: file && !loading ? "pointer" : "not-allowed",
                transition: "all 0.3s",
                letterSpacing: 1
              }}
            >
              {loading ? "⏳ Analyzing..." : "🔍 Analyze Video"}
            </button>

            {/* ERROR */}
            {error && (
              <div style={{
                marginTop: 16,
                padding: 16,
                background: "rgba(248,113,113,0.1)",
                border: "1px solid rgba(248,113,113,0.3)",
                borderRadius: 12,
                color: "#f87171",
                fontSize: 14
              }}>
                ❌ {error}
              </div>
            )}

            {/* HOW IT WORKS */}
            {!result && !loading && (
              <div style={{
                marginTop: 32,
                padding: 24,
                background: "rgba(15,23,42,0.6)",
                border: "1px solid #1e293b",
                borderRadius: 16
              }}>
                <h3 style={{
                  color: "#94a3b8",
                  fontSize: 13,
                  letterSpacing: 2,
                  marginBottom: 20,
                  marginTop: 0
                }}>
                  HOW IT WORKS
                </h3>
                {[
                  { icon: "🎬", step: "01", title: "Upload Video", desc: "Upload any MP4, AVI, MOV or MKV file" },
                  { icon: "🔍", step: "02", title: "Frame Analysis", desc: "ViT model analyzes every frame" },
                  { icon: "🗳️", step: "03", title: "Majority Voting", desc: "Combines all frame predictions" },
                  { icon: "🔥", step: "04", title: "Grad-CAM", desc: "Highlights suspicious face regions" }
                ].map((s, i) => (
                  <div key={i} style={{
                    display: "flex",
                    gap: 16,
                    marginBottom: 16,
                    alignItems: "flex-start"
                  }}>
                    <div style={{
                      width: 36,
                      height: 36,
                      borderRadius: 10,
                      background: "rgba(0,212,255,0.1)",
                      border: "1px solid rgba(0,212,255,0.2)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 18,
                      flexShrink: 0
                    }}>
                      {s.icon}
                    </div>
                    <div>
                      <div style={{
                        fontSize: 14,
                        fontWeight: 600,
                        color: "#e2e8f0"
                      }}>
                        {s.title}
                      </div>
                      <div style={{ fontSize: 12, color: "#475569" }}>
                        {s.desc}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* RIGHT: RESULT */}
          {loading && (
            <div style={{
              background: "rgba(15,23,42,0.8)",
              border: "1px solid #1e293b",
              borderRadius: 20,
              padding: 48,
              textAlign: "center",
              backdropFilter: "blur(10px)"
            }}>
              <div style={{ fontSize: 64, marginBottom: 24 }}>⏳</div>
              <h3 style={{ color: "#00d4ff", marginBottom: 12 }}>
                Analyzing Video
              </h3>
              <p style={{ color: "#64748b", marginBottom: 8 }}>
                Extracting and analyzing frames...
              </p>
              <p style={{ color: "#475569", fontSize: 13 }}>
                Running ViT model + Grad-CAM on each frame
              </p>
              <div style={{
                marginTop: 32,
                display: "flex",
                flexDirection: "column",
                gap: 10
              }}>
                {[
                  "Extracting frames with OpenCV",
                  "Running ViT model on each frame",
                  "Calculating majority vote",
                  "Generating Grad-CAM heatmap"
                ].map((step, i) => (
                  <div key={i} style={{
                    padding: "10px 16px",
                    background: "rgba(0,212,255,0.05)",
                    border: "1px solid rgba(0,212,255,0.1)",
                    borderRadius: 8,
                    fontSize: 13,
                    color: "#64748b",
                    textAlign: "left",
                    display: "flex",
                    alignItems: "center",
                    gap: 10
                  }}>
                    <span style={{ color: "#00d4ff" }}>⚙️</span> {step}
                  </div>
                ))}
              </div>
            </div>
          )}

          {result && (
            <div style={{
              background: "rgba(15,23,42,0.8)",
              border: `2px solid ${result.verdict === "FAKE"
                ? "rgba(248,113,113,0.5)"
                : "rgba(52,211,153,0.5)"}`,
              borderRadius: 20,
              padding: 32,
              backdropFilter: "blur(10px)"
            }}>

              {/* VERDICT */}
              <div style={{
                textAlign: "center",
                marginBottom: 28,
                padding: "24px",
                background: result.verdict === "FAKE"
                  ? "rgba(248,113,113,0.08)"
                  : "rgba(52,211,153,0.08)",
                borderRadius: 16
              }}>
                <div style={{ fontSize: 56, marginBottom: 8 }}>
                  {result.verdict === "FAKE" ? "🔴" : "🟢"}
                </div>
                <h2 style={{
                  fontSize: 40,
                  fontWeight: 900,
                  margin: "0 0 4px",
                  color: result.verdict === "FAKE" ? "#f87171" : "#34d399"
                }}>
                  {result.verdict}
                </h2>
                <p style={{ color: "#64748b", fontSize: 13, margin: 0 }}>
                  {result.filename}
                </p>
              </div>

              {/* STATS */}
              <div style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 12,
                marginBottom: 24
              }}>
                {[
                  { label: "Confidence", value: `${result.confidence.toFixed(1)}%`, color: "#00d4ff", icon: "🎯" },
                  { label: "Fake Frames", value: `${result.fake_percent.toFixed(1)}%`, color: "#f87171", icon: "🎞️" }
                ].map((s, i) => (
                  <div key={i} style={{
                    padding: 20,
                    background: "#0f172a",
                    borderRadius: 12,
                    textAlign: "center",
                    border: "1px solid #1e293b"
                  }}>
                    <div style={{ fontSize: 24, marginBottom: 4 }}>{s.icon}</div>
                    <div style={{
                      fontSize: 28,
                      fontWeight: 900,
                      color: s.color
                    }}>
                      {s.value}
                    </div>
                    <div style={{ fontSize: 12, color: "#475569" }}>
                      {s.label}
                    </div>
                  </div>
                ))}
              </div>

              {/* IMAGES */}
              <div style={{ marginBottom: 16 }}>
                <p style={{
                  color: "#475569",
                  fontSize: 11,
                  letterSpacing: 2,
                  marginBottom: 8,
                  marginTop: 0
                }}>
                  GRAD-CAM — SUSPICIOUS REGIONS
                </p>
                <img
                  src={`${API_URL}${result.heatmap_url}?t=${Date.now()}`}
                  alt="Grad-CAM"
                  style={{
                    width: "100%",
                    borderRadius: 12,
                    border: "1px solid #1e293b"
                  }}
                />
              </div>

              <div>
                <p style={{
                  color: "#475569",
                  fontSize: 11,
                  letterSpacing: 2,
                  marginBottom: 8,
                  marginTop: 0
                }}>
                  FRAME-BY-FRAME ANALYSIS
                </p>
                <img
                  src={`${API_URL}${result.graph_url}?t=${Date.now()}`}
                  alt="Graph"
                  style={{
                    width: "100%",
                    borderRadius: 12,
                    border: "1px solid #1e293b"
                  }}
                />
              </div>

              <div style={{
                marginTop: 16,
                textAlign: "center",
                color: "#334155",
                fontSize: 11
              }}>
                ID: {result.id} • Analyzed with DeepShield v1.0
              </div>
            </div>
          )}
        </div>

        {/* HISTORY */}
        {showHistory && (
          <div style={{
            marginTop: 48,
            background: "rgba(15,23,42,0.8)",
            border: "1px solid #1e293b",
            borderRadius: 20,
            padding: 32
          }}>
            <h3 style={{
              color: "#94a3b8",
              fontSize: 13,
              letterSpacing: 2,
              marginTop: 0,
              marginBottom: 20
            }}>
              ANALYSIS HISTORY ({history.length})
            </h3>

            {history.length === 0 ? (
              <p style={{ color: "#475569", textAlign: "center" }}>
                No analyses yet.
              </p>
            ) : (
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
                gap: 12
              }}>
                {history.map((item, i) => (
                  <div key={i} style={{
                    padding: "16px 20px",
                    background: "#0f172a",
                    border: `1px solid ${item.verdict === "FAKE"
                      ? "rgba(248,113,113,0.2)"
                      : "rgba(52,211,153,0.2)"}`,
                    borderRadius: 12,
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center"
                  }}>
                    <div>
                      <div style={{
                        color: "#e2e8f0",
                        fontSize: 14,
                        fontWeight: 600,
                        marginBottom: 4
                      }}>
                        {item.filename}
                      </div>
                      <div style={{ color: "#475569", fontSize: 11 }}>
                        {item.created_at}
                      </div>
                      <div style={{ color: "#64748b", fontSize: 11 }}>
                        Confidence: {item.confidence}%
                      </div>
                    </div>
                    <div style={{
                      fontSize: 22,
                      fontWeight: 900,
                      color: item.verdict === "FAKE" ? "#f87171" : "#34d399"
                    }}>
                      {item.verdict === "FAKE" ? "🔴" : "🟢"}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}