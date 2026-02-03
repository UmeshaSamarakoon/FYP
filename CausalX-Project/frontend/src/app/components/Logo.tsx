export function Logo() {
  return (
    <div className="flex items-center gap-2">
      <div className="relative w-10 h-10">
        {/* Circular background */}
        <svg viewBox="0 0 40 40" className="w-full h-full">
          {/* Gradient definition */}
          <defs>
            <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="hsl(var(--primary))" />
              <stop offset="100%" stopColor="hsl(var(--destructive))" />
            </linearGradient>
          </defs>
          
          {/* Circle */}
          <circle cx="20" cy="20" r="18" fill="url(#logoGradient)" />
          
          {/* Causal link symbol - connected nodes */}
          <circle cx="13" cy="15" r="3" fill="white" />
          <circle cx="27" cy="15" r="3" fill="white" />
          <circle cx="20" cy="27" r="3" fill="white" />
          
          {/* Connection lines */}
          <line x1="13" y1="15" x2="27" y2="15" stroke="white" strokeWidth="2" />
          <line x1="13" y1="15" x2="20" y2="27" stroke="white" strokeWidth="2" />
          <line x1="27" y1="15" x2="20" y2="27" stroke="white" strokeWidth="2" />
          
          {/* X mark overlay for "break" indication */}
          <line x1="18" y1="18" x2="22" y2="22" stroke="hsl(var(--destructive))" strokeWidth="2.5" strokeLinecap="round" />
          <line x1="22" y1="18" x2="18" y2="22" stroke="hsl(var(--destructive))" strokeWidth="2.5" strokeLinecap="round" />
        </svg>
      </div>
      <div className="flex flex-col leading-none">
        <span className="text-xl font-bold">CausalX</span>
        <span className="text-xs text-muted-foreground">Deepfake Detection</span>
      </div>
    </div>
  );
}
