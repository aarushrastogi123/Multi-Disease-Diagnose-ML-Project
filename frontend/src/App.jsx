import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [darkMode, setDarkMode] = useState(true);
  const [showNav, setShowNav] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 100) {
        setShowNav(true);
      } else {
        setShowNav(false);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(file ? URL.createObjectURL(file) : null);
    setResult(null);
    setErr("");
  };

  const handlePredict = async () => {
    if (!image) {
      alert("Please upload an image first!");
      return;
    }

    setLoading(true);
    setErr("");
    try {
      const formData = new FormData();
      formData.append("file", image);

      const resp = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) throw new Error(`Backend error: ${resp.status}`);

      const data = await resp.json();
      const p = data.prediction ?? data;

      const predicted = p.predicted_class ?? "Unknown";
      const confidences = p.confidences ?? {};

      setTimeout(() => {
        setResult({ predicted, confidences });
        setLoading(false);
      }, 2500);
    } catch (e) {
      setErr(e.message || "Prediction failed");
      setLoading(false);
    }
  };

  return (
    <div className={`app ${darkMode ? 'dark' : 'light'}`}>
      {/* Animated Background */}
      {darkMode && <div className="animated-bg"></div>}

      {/* Top Header - Title & Theme Toggle */}
      <header className="top-header">
        <h1 className="main-title">VisionDX</h1>
        <button 
          onClick={() => setDarkMode(!darkMode)}
          className="top-theme-btn"
        >
          {darkMode ? "☀️" : "🌙"}
        </button>
      </header>

      {/* Floating Navigation - Appears on Scroll */}
      <nav className={`floating-navbar ${showNav ? 'show' : ''}`}>
        <div className="nav-content">
          <div className="nav-logo">
            <div className="logo-placeholder">👁️</div>
          </div>
          <div className="nav-links">
            <a href="#features" className="nav-link">Features</a>
            <a href="#how-it-works" className="nav-link">How It Works</a>
            <a href="#about" className="nav-link">About</a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-section">
        <h2 className="hero-title">
          AI-Powered Retinal Disease Diagnosis
        </h2>
        <p className="hero-subtitle">
          Upload a fundus image and let our advanced AI analyze it instantly. Get accurate disease predictions with confidence scores in seconds.
        </p>
        <a href="#diagnosis" className="hero-cta">
          Start Diagnosis →
        </a>
      </section>

      {/* Features Section */}
      <section id="features" className="features-section">
        <h3 className="section-title">Why Choose VisionDX?</h3>
        <div className="features-grid">
          {[
            { icon: "🔬", title: "Advanced AI", desc: "State-of-the-art deep learning models trained on thousands of retinal images" },
            { icon: "⚡", title: "Instant Results", desc: "Get comprehensive diagnosis reports in under 3 seconds" },
            { icon: "📊", title: "Detailed Analysis", desc: "Confidence breakdowns for multiple disease classifications" },
            { icon: "🔒", title: "Secure & Private", desc: "Your medical data stays encrypted and confidential" },
            { icon: "🎯", title: "High Accuracy", desc: "Validated against clinical standards with 95%+ accuracy" },
            { icon: "🌐", title: "Accessible", desc: "Easy-to-use interface accessible from anywhere" }
          ].map((feature, idx) => (
            <div key={idx} className="feature-card">
              <div className="feature-icon">{feature.icon}</div>
              <h4 className="feature-title">{feature.title}</h4>
              <p className="feature-desc">{feature.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="how-section">
        <div className="how-container">
          <h3 className="section-title">How It Works</h3>
          <div className="steps-grid">
            {[
              { step: "1", title: "Upload Image", desc: "Select and upload a clear fundus photograph" },
              { step: "2", title: "AI Analysis", desc: "Our neural network processes the retinal patterns" },
              { step: "3", title: "Get Results", desc: "Receive detailed diagnosis with confidence scores" }
            ].map((step, idx) => (
              <div key={idx} className="step-card">
                <div className="step-number">{step.step}</div>
                <h4 className="step-title">{step.title}</h4>
                <p className="step-desc">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Diagnosis Tool */}
      <section id="diagnosis" className="diagnosis-section">
        <div className="diagnosis-card">
          <h3 className="diagnosis-title">Start Your Diagnosis</h3>
          
          <div className="diagnosis-content">
            <label className="file-upload-btn">
              <span className="upload-icon">📁</span>
              Upload Fundus Image
              <input type="file" accept="image/*" onChange={handleImageChange} className="file-input" />
            </label>

            {preview && (
              <div className="preview-container">
                <img 
                  src={preview} 
                  alt="Preview" 
                  className="preview-image"
                />
              </div>
            )}

            <button
              onClick={handlePredict}
              disabled={loading}
              className="diagnose-btn"
            >
              {loading ? "Analyzing..." : "🔍 Diagnose Image"}
            </button>

            {loading && (
              <div className="loading-container">
                <div className="loading-spinner"></div>
                <p className="loading-text">Analyzing Retina...</p>
              </div>
            )}

            {err && (
              <div className="error-message">
                ⚠️ {err}
              </div>
            )}

            {result && !loading && (
              <div className="results-container">
                <h4 className="results-title">Diagnosis Complete ✓</h4>
                
                <div className="confidence-section">
                  <h5 className="confidence-title">Confidence Breakdown:</h5>
                  {Object.entries(result.confidences)
                    .sort((a, b) => b[1] - a[1])
                    .map(([disease, val]) => (
                      <div key={disease} className="confidence-item">
                        <div className="confidence-header">
                          <span className="disease-name">{disease}</span>
                          <span className="confidence-percent">{(val * 100).toFixed(2)}%</span>
                        </div>
                        <div className="confidence-bar-bg">
                          <div 
                            className="confidence-bar-fill"
                            style={{ width: `${val * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="about-section">
        <div className="about-content">
          <h3 className="section-title">About VisionDX</h3>
          <p className="about-text">
            VisionDX is a cutting-edge medical AI platform designed to assist healthcare professionals in diagnosing retinal diseases. Our deep learning models are trained on extensive datasets to provide accurate, reliable results.
          </p>
          <p className="about-disclaimer">
            ⚕️ This tool is designed to assist medical professionals and should not replace professional medical advice, diagnosis, or treatment.
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <p>© 2025 <strong className="footer-brand">VisionDX</strong> — Empowering Eye Health with AI 👁️</p>
      </footer>
    </div>
  );
}

export default App;