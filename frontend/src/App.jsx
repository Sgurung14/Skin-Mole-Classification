import { useEffect, useState } from "react";

const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setResult(null);

    if (!file) {
      setError("Select an image first.");
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${DEFAULT_API_BASE.replace(/\/$/, "")}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || "Prediction request failed.");
      }
      setResult(data);
    } catch (err) {
      setError(err.message || "Unexpected error.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="panel">
        <header className="hero">
          <p className="eyebrow">Skin Mole Classification</p>
          <h1>Upload an image to get a prediction</h1>
          <p className="subtle">
            React UI for the FastAPI <code>/predict</code> endpoint.
          </p>
        </header>

        <form onSubmit={handleSubmit} className="form-grid">
          <label className="upload-card">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <span>{file ? file.name : "Choose image (jpg/png/...)"}</span>
          </label>

          <button type="submit" disabled={loading}>
            {loading ? "Predicting..." : "Run Prediction"}
          </button>
        </form>

        {error ? <p className="error">{error}</p> : null}
        <p className="subtle api-target">
          API: <code>{DEFAULT_API_BASE}</code>
        </p>

        <div className="content-grid">
          <section className="card">
            <h2>Image Preview</h2>
            {previewUrl ? (
              <img src={previewUrl} alt="Selected skin lesion" className="preview" />
            ) : (
              <p className="placeholder">No image selected.</p>
            )}
          </section>

          <section className="card">
            <h2>Prediction Output</h2>
            {result ? (
              <dl className="result-list">
                <div>
                  <dt>Label</dt>
                  <dd className={result.pred_label === "malignant" ? "danger" : "ok"}>
                    {result.pred_label}
                  </dd>
                </div>
                <div>
                  <dt>Malignant Probability</dt>
                  <dd>{(Number(result.prob_malignant) * 100).toFixed(2)}%</dd>
                </div>
                <div>
                  <dt>Threshold</dt>
                  <dd>{Number(result.threshold).toFixed(2)}</dd>
                </div>
              </dl>
            ) : (
              <p className="placeholder">Prediction results will appear here.</p>
            )}
          </section>
        </div>
      </section>
    </main>
  );
}

export default App;
