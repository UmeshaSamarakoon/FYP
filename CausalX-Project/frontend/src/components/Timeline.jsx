export default function Timeline({ frames, onSelect }) {
  return (
    <div style={{ marginTop: "20px" }}>
      <h4>Causal Timeline</h4>
      {frames.map((f, i) => (
        <div
          key={i}
          onClick={() => onSelect(f.timestamp)}
          style={{
            cursor: "pointer",
            color: f.fake_prob > 0.6 ? "red" : "black"
          }}
        >
          ⏱ {f.timestamp.toFixed(2)}s — FakeProb: {f.fake_prob.toFixed(2)}
        </div>
      ))}
    </div>
  );
}
