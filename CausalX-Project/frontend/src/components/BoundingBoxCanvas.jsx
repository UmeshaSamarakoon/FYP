import { useEffect, useRef } from "react";

export default function BoundingBoxCanvas({ frame, videoWidth, videoHeight }) {
  const canvasRef = useRef();

  useEffect(() => {
    if (!frame || !frame.bbox) return;

    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, videoWidth, videoHeight);

    const [x1, y1, x2, y2] = frame.bbox;

    ctx.strokeStyle = frame.fake_prob > 0.6 ? "red" : "green";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }, [frame]);

  return (
    <canvas
      ref={canvasRef}
      width={videoWidth}
      height={videoHeight}
      style={{ position: "absolute", top: 0, left: 0 }}
    />
  );
}
