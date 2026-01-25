import { useEffect, useRef } from "react";

export default function BoundingBoxCanvas({
  frame,
  videoWidth,
  videoHeight,
  sourceWidth,
  sourceHeight
}) {
  const canvasRef = useRef();

  useEffect(() => {
    if (!frame || !frame.bbox) return;

    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, videoWidth, videoHeight);

    const [x1, y1, x2, y2] = frame.bbox;
    const scaleX = sourceWidth ? videoWidth / sourceWidth : 1;
    const scaleY = sourceHeight ? videoHeight / sourceHeight : 1;
    const drawX = x1 * scaleX;
    const drawY = y1 * scaleY;
    const drawW = (x2 - x1) * scaleX;
    const drawH = (y2 - y1) * scaleY;

    ctx.strokeStyle = frame.fake_prob > 0.6 ? "red" : "green";
    ctx.lineWidth = 3;
    ctx.strokeRect(drawX, drawY, drawW, drawH);
  }, [frame, videoWidth, videoHeight, sourceWidth, sourceHeight]);

  return (
    <canvas
      ref={canvasRef}
      width={videoWidth}
      height={videoHeight}
      style={{ position: "absolute", top: 0, left: 0 }}
    />
  );
}
