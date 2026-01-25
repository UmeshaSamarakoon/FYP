export default function Timeline({ segments, totalDuration, onSelect }) {
  if (!totalDuration) {
    return null;
  }

  return (
    <div className="timeline">
      <h4>Lip-Sync Break Timeline</h4>
      <div className="timeline-bar">
        {segments.map((segment, i) => {
          const [start, end] = segment;
          const left = (start / totalDuration) * 100;
          const width = ((end - start) / totalDuration) * 100;
          return (
            <button
              key={`${start}-${end}-${i}`}
              type="button"
              className="timeline-segment"
              style={{ left: `${left}%`, width: `${width}%` }}
              onClick={() => onSelect(start)}
            />
          );
        })}
      </div>
    </div>
  );
}
