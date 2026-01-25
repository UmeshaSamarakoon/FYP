import { useRef, useEffect } from "react";

export default function VideoPlayer({ videoSrc, currentTime, onTimeUpdate }) {
  const videoRef = useRef();

  useEffect(() => {
    if (videoRef.current && currentTime !== null) {
      videoRef.current.currentTime = currentTime;
    }
  }, [currentTime]);

  return (
    <video
      ref={videoRef}
      src={videoSrc}
      controls
      width="720"
      onTimeUpdate={(e) => onTimeUpdate(e.target.currentTime)}
    />
  );
}
