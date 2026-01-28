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
  const [videoMeta, setVideoMeta] = useState({ width: 0, height: 0, duration: 0 });
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

  const segments = useMemo(() => {
    if (!frames.length && !summary?.highlights?.length) return [];
    const sorted = [...frames].sort((a, b) => a.timestamp - b.timestamp);
    const deltas = [];
    for (let i = 1; i < Math.min(sorted.length, 6); i += 1) {
      deltas.push(sorted[i].timestamp - sorted[i - 1].timestamp);
    }
    const frameStep = deltas.length
      ? deltas.reduce((a, b) => a + b, 0) / deltas.length
      : 0.04;
    const maxGap = frameStep * 1.5;
    const highlightTimes = summary?.highlights?.length ? summary.highlights : null;
    const flagged = highlightTimes
      ? highlightTimes.map((timestamp) => ({ timestamp }))
      : sorted.filter(f => f.fake_prob >= probThreshold);
    if (!flagged.length) return [];

    const ranges = [];
    let start = flagged[0].timestamp;
    let last = flagged[0].timestamp;

    for (let i = 1; i < flagged.length; i += 1) {
      const t = flagged[i].timestamp;
      if (t - last > maxGap) {
        ranges.push([start, last + frameStep]);
        start = t;
      }
      last = t;
    }
    ranges.push([start, last + frameStep]);
    return ranges.map(([s, e]) => [
      Math.max(0, s),
      Math.min(videoMeta.duration || e, e)
    ]);
  }, [frames, probThreshold, summary?.highlights, videoMeta.duration]);

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
            style={{ height: `${videoDims.height}px` }}
            onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)}
            onLoadedMetadata={(e) => {
              const videoWidth = e.target.videoWidth || 720;
              const videoHeight = e.target.videoHeight || 405;
              const duration = e.target.duration || 0;
              const scale = 720 / videoWidth;
              setVideoMeta({ width: videoWidth, height: videoHeight, duration });
              setVideoDims({ width: 720, height: Math.round(videoHeight * scale) });
            }}
          />

          {active?.bbox && isFake && active.fake_prob >= probThreshold && (
            <BoundingBoxCanvas
              frame={active}
              videoWidth={videoDims.width}
              videoHeight={videoDims.height}
              sourceWidth={videoMeta.width}
              sourceHeight={videoMeta.height}
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
        <Timeline
          segments={segments}
          totalDuration={videoMeta.duration}
          onSelect={(t) => {
            if (videoRef.current) {
              videoRef.current.currentTime = t;
            }
            setCurrentTime(t);
          }}
        />
      )}
    </div>
  );
}
