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
  const [sourceDims, setSourceDims] = useState({ width: 720, height: 405 });
  const [videoDuration, setVideoDuration] = useState(0);
  const [causalSegments, setCausalSegments] = useState([]);
  const videoRef = useRef(null);
  const probThreshold = 0.6;

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    setStatus("Analyzing video...");
    setVideoURL(URL.createObjectURL(file));
    setFrames([]);
    setSummary(null);
    setCausalSegments([]);

    try {
      const result = await analyzeVideo(file);
      setFrames(result.frames || []);
      setSummary({
        label: result.video_fake,
        confidence: result.fake_confidence,
        overallScore: result.overall_score,
        highlights: result.highlight_timestamps || []
      });
      setCausalSegments(result.causal_segments || []);
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
  const limitedSegments = causalSegments.slice(0, 5);

  return (
    <div className="container">
      <h1>Causal Visualization Interface (CVI)</h1>

      <input type="file" accept="video/mp4" onChange={handleUpload} />
      <p>{status}</p>

      {summary && (
        <div className="panel">
          <p><b>Video verdict:</b> {summary.label}</p>
          <p><b>Overall video score:</b> {summary.overallScore?.toFixed(3) ?? "N/A"}</p>
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
              setSourceDims({ width: videoWidth, height: videoHeight });
              setVideoDuration(e.target.duration || 0);
            }}
          />

          {active?.bbox && isFake && active.fake_prob >= probThreshold && (
            <BoundingBoxCanvas
              frame={active}
              videoWidth={videoDims.width}
              videoHeight={videoDims.height}
              sourceWidth={sourceDims.width}
              sourceHeight={sourceDims.height}
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

      {videoURL && (
        <>
          <Timeline
            segments={causalSegments}
            totalDuration={videoDuration}
            onSelect={(t) => {
              if (videoRef.current) {
                videoRef.current.currentTime = t;
              }
              setCurrentTime(t);
            }}
          />

          {causalSegments.length > 0 && (
            <div className="panel">
              <h3>Detected Causal Link Breaks</h3>
              <p><b>Total segments:</b> {causalSegments.length}</p>
              {limitedSegments.map((segment, i) => (
                <p key={`${segment[0]}-${segment[1]}-${i}`} className="fake-timestamp">
                  ⛔ {segment[0].toFixed(2)}s – {segment[1].toFixed(2)}s
                </p>
              ))}
              {causalSegments.length > limitedSegments.length && (
                <p>Showing first {limitedSegments.length} segments.</p>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
