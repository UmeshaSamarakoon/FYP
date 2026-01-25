import { useState, useRef } from "react";
import { analyzeVideo } from "./api";

export default function App() {
  const [frames, setFrames] = useState([]);
  const [videoURL, setVideoURL] = useState(null);
  const [status, setStatus] = useState("");
  const videoRef = useRef(null);

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    setStatus("Analyzing video...");
    setVideoURL(URL.createObjectURL(file));

    try {
      const result = await analyzeVideo(file);
      setFrames(result.frames);
      setStatus("Analysis complete");
    } catch (err) {
      setStatus("Analysis failed");
    }
  }

  function getActiveFrame() {
    if (!videoRef.current) return null;
    const t = videoRef.current.currentTime;
    return frames.find(f => Math.abs(f.timestamp - t) < 0.05);
  }

  const active = getActiveFrame();

  return (
    <div className="container">
      <h1>Causal Visualization Interface (CVI)</h1>

      <input type="file" accept="video/mp4" onChange={handleUpload} />
      <p>{status}</p>

      {videoURL && (
        <div className="video-wrapper">
          <video
            ref={videoRef}
            src={videoURL}
            controls
            width="720"
          />

          {active?.bbox && (
            <div
              className="bbox"
              style={{
                left: active.bbox[0],
                top: active.bbox[1],
                width: active.bbox[2] - active.bbox[0],
                height: active.bbox[3] - active.bbox[1]
              }}
            />
          )}
        </div>
      )}

      {active && (
        <div className="panel">
          <p><b>Timestamp:</b> {active.timestamp.toFixed(2)} s</p>
          <p><b>Fake probability:</b> {active.fake_prob.toFixed(3)}</p>
          <p><b>AV mismatch:</b> {active.av_mismatch.toFixed(3)}</p>
        </div>
      )}
    </div>
  );

  const fakeFrames = frames.filter(f => f.fake_prob > 0.6);

  {fakeFrames.length > 0 && (
  <div className="panel">
    <h3>Detected Manipulation Segments</h3>
    {fakeFrames.map((f, i) => (
      <p key={i}>
        ⛔ {f.timestamp.toFixed(2)}s — prob {f.fake_prob.toFixed(2)}
      </p>
    ))}
  </div>
)}

}
