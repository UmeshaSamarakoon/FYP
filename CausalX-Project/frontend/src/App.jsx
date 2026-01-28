import { useMemo, useState, useRef } from "react";
import { analyzeVideo } from "./api";
import BoundingBoxCanvas from "./components/BoundingBoxCanvas";
import Timeline from "./components/Timeline";

export default function App() {
  const [frames, setFrames] = useState([]);
  const [videoURL, setVideoURL] = useState(null);
  const [status, setStatus] = useState("");
  const [summary, setSummary] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDims, setVideoDims] = useState({ width: 720, height: 405 });
  const videoRef = useRef(null);
  const probThreshold = 0.6;

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    setStatus("Analyzing video...");
    setVideoURL(URL.createObjectURL(file));
    setFrames([]);
    setSummary(null);

    try {
      const result = await analyzeVideo(file);
      setFrames(result.frames || []);
      setSummary({
        label: result.video_fake,
        confidence: result.fake_confidence,
        highlights: result.highlight_timestamps || []
      });
      setStatus("Analysis complete");
    } catch (err) {
      setStatus("Analysis failed");
    }
  }

  function getActiveFrame() {
    if (!frames.length) return null;
    const t = currentTime;
    return frames.find(f => Math.abs(f.timestamp - t) < 0.05);
  }

  const active = getActiveFrame();
  const isFake = summary?.label === "FAKE";
  const fakeFrames = isFake ? frames.filter(f => f.fake_prob >= probThreshold) : [];

  return (
    <div className="container">
      <h1>Causal Visualization Interface (CVI)</h1>

      <input type="file" accept="video/mp4" onChange={handleUpload} />
      <p>{status}</p>

      {summary && (
        <div className="panel">
          <p><b>Video verdict:</b> {summary.label}</p>
          <p><b>Fake frame ratio:</b> {summary.confidence?.toFixed(3) ?? "N/A"}</p>
        </div>
      )}

      {videoURL && (
        <div className="video-wrapper">
          <video
            ref={videoRef}
            src={videoURL}
            controls
            width="720"
            onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)}
            onLoadedMetadata={(e) => {
              const videoWidth = e.target.videoWidth || 720;
              const videoHeight = e.target.videoHeight || 405;
              const scale = 720 / videoWidth;
              setVideoDims({ width: 720, height: Math.round(videoHeight * scale) });
            }}
          />

          {active?.bbox && isFake && active.fake_prob >= probThreshold && (
            <BoundingBoxCanvas
              frame={active}
              videoWidth={videoDims.width}
              videoHeight={videoDims.height}
            />
          )}
        </div>
      )}

      {active && (
        <div className="panel">
          <p><b>Fake probability:</b> {active.fake_prob.toFixed(3)}</p>
          <p><b>AV mismatch:</b> {active.av_mismatch.toFixed(3)}</p>
        </div>
      )}

      {isFake && (
        <>
          <Timeline
            frames={frames}
            onSelect={(t) => {
              if (videoRef.current) {
                videoRef.current.currentTime = t;
              }
              setCurrentTime(t);
            }}
          />

          {fakeFrames.length > 0 && (
            <div className="panel">
              <h3>Detected Lip-Sync Breaks</h3>
              {fakeFrames.map((f, i) => (
                <p key={i} className="fake-timestamp">
                  ⛔ {f.timestamp.toFixed(2)}s — prob {f.fake_prob.toFixed(2)}
                </p>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
