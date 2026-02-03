import { useMemo, useRef, useState, useEffect } from 'react';
import { Play, Pause, SkipBack, SkipForward, Volume2, VolumeX, Maximize } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Badge } from '@/app/components/ui/badge';
import type { FrameResult } from '@/app/lib/api';

interface CausalBreachSegment {
  start: number;
  end: number;
  score: number;
}

interface VideoAnalysisProps {
  videoFile: File;
  result: 'REAL' | 'FAKE';
  confidence: number;
  breachSegments: CausalBreachSegment[];
  frames: FrameResult[];
  probThreshold?: number;
}

export function VideoAnalysis({ videoFile, result, confidence, breachSegments, frames, probThreshold = 0.6 }: VideoAnalysisProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const progressBarRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string>('');
  const [naturalSize, setNaturalSize] = useState({ width: 1, height: 1 });

  useEffect(() => {
    const url = URL.createObjectURL(videoFile);
    setVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => setCurrentTime(video.currentTime);
    const handleLoadedMetadata = () => {
      setDuration(video.duration);
      setNaturalSize({
        width: video.videoWidth || 1,
        height: video.videoHeight || 1,
      });
    };
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => setIsPlaying(false);

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('ended', handleEnded);

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('ended', handleEnded);
    };
  }, []);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play();
    }
  };

  const handleProgressBarClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!progressBarRef.current || !videoRef.current) return;
    const rect = progressBarRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = x / rect.width;
    const newTime = percent * duration;
    videoRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const handleVolumeClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!videoRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = x / rect.width;
    const newVolume = Math.max(0, Math.min(1, percent));
    videoRef.current.volume = newVolume;
    setVolume(newVolume);
    setIsMuted(newVolume === 0);
  };

  const toggleMute = () => {
    if (!videoRef.current) return;
    const newMuted = !isMuted;
    videoRef.current.muted = newMuted;
    setIsMuted(newMuted);
  };

  const skipTime = (seconds: number) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Math.max(0, Math.min(duration, currentTime + seconds));
  };

  const toggleFullscreen = () => {
    if (!videoRef.current) return;
    const videoContainer = videoRef.current.parentElement;
    if (!videoContainer) return;
    
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      videoContainer.requestFullscreen();
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const isInBreach = (time: number) => {
    return breachSegments.some(seg => time >= seg.start && time <= seg.end);
  };

  const activeFrame = useMemo(() => {
    if (!frames.length) return null;
    let closest = frames[0];
    let minDiff = Math.abs(currentTime - closest.timestamp);
    for (const f of frames) {
      const diff = Math.abs(currentTime - f.timestamp);
      if (diff < minDiff) {
        closest = f;
        minDiff = diff;
      }
    }
    return closest;
  }, [frames, currentTime]);

  const showBoundingBox =
    result === 'FAKE' &&
    activeFrame?.bbox &&
    activeFrame.fake_prob >= probThreshold;

  const currentBreach = breachSegments.find(seg => currentTime >= seg.start && currentTime <= seg.end);
  const breachScore = currentBreach?.score;

  const isFake = result === 'FAKE';
  const bbox = activeFrame?.bbox;
  const bboxStyle = useMemo(() => {
    if (!bbox || !videoRef.current) return null;
    const [x1, y1, x2, y2] = bbox;
    const vw = videoRef.current.clientWidth || naturalSize.width;
    const vh = videoRef.current.clientHeight || naturalSize.height;
    const scaleX = vw / naturalSize.width;
    const scaleY = vh / naturalSize.height;
    return {
      left: `${(x1 * scaleX)}px`,
      top: `${(y1 * scaleY)}px`,
      width: `${(x2 - x1) * scaleX}px`,
      height: `${(y2 - y1) * scaleY}px`,
    };
  }, [bbox, naturalSize]);

  return (
    <div className="w-full max-w-6xl space-y-6">
      {/* Result Badge */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Badge 
            variant={isFake ? "destructive" : "default"}
            className={`text-xl px-6 py-2 ${!isFake && 'bg-green-500 hover:bg-green-600'}`}
          >
            {result}
          </Badge>
          <span className="text-lg text-muted-foreground">
            Confidence: {confidence.toFixed(1)}%
          </span>
        </div>
        <p className="text-muted-foreground">{videoFile.name}</p>
      </div>

      {/* Video Player with Bounding Box */}
      <div className="relative bg-black rounded-lg overflow-hidden">
        <video
          ref={videoRef}
          src={videoUrl}
          className="w-full aspect-video"
          onClick={togglePlay}
        />
        
        {/* Bounding Box Overlay - only for FAKE videos */}
        {showBoundingBox && bboxStyle && (
          <div className="absolute inset-0 pointer-events-none">
            <div
              className="absolute border-4 border-red-500 shadow-[0_0_20px_rgba(239,68,68,0.8)] animate-pulse"
              style={bboxStyle}
            >
              <div className="absolute -top-2 -left-2 w-4 h-4 bg-red-500 rounded-full" />
              <div className="absolute -top-2 -right-2 w-4 h-4 bg-red-500 rounded-full" />
              <div className="absolute -bottom-2 -left-2 w-4 h-4 bg-red-500 rounded-full" />
              <div className="absolute -bottom-2 -right-2 w-4 h-4 bg-red-500 rounded-full" />
              <div className="absolute -top-10 left-1/2 -translate-x-1/2 bg-red-500 text-white px-4 py-2 rounded text-sm font-bold whitespace-nowrap shadow-lg">
                CAUSAL BREACH {breachScore && `(${breachScore.toFixed(2)})`}
              </div>
            </div>
          </div>
        )}

        {/* Breach Indicator - only for FAKE videos */}
        {showBoundingBox && (
          <div className="absolute top-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg font-bold text-lg shadow-lg animate-pulse">
            LIP-SYNC MISMATCH DETECTED
          </div>
        )}
        
        {/* Clean indicator for REAL videos */}
        {!isFake && (
          <div className="absolute top-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg font-bold text-lg shadow-lg">
            NO ANOMALIES DETECTED
          </div>
        )}
      </div>

      {/* Integrated Timeline & Controls */}
      <div className="space-y-4">
        {/* Progress Bar with Breach Segments */}
        <div 
          ref={progressBarRef}
          className="relative h-2 bg-black rounded-full overflow-hidden cursor-pointer group"
          onClick={handleProgressBarClick}
        >
          {/* Breach Segments Background - only show for FAKE videos */}
          {isFake && breachSegments.map((segment, idx) => {
            const left = (segment.start / duration) * 100;
            const width = ((segment.end - segment.start) / duration) * 100;
            
            return (
              <div
                key={idx}
                className="absolute top-0 bottom-0 bg-red-500 z-10"
                style={{
                  left: `${left}%`,
                  width: `${width}%`
                }}
              />
            );
          })}
          
          {/* Progress Bar */}
          <div
            className="absolute top-0 bottom-0 bg-primary transition-all z-20"
            style={{ width: `${(currentTime / duration) * 100}%` }}
          />
          
          {/* Playhead */}
          <div
            className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-primary rounded-full shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-30"
            style={{ left: `${(currentTime / duration) * 100}%`, marginLeft: '-8px' }}
          />
        </div>
        
        {/* Time Display */}
        <div className="flex justify-between text-sm text-muted-foreground px-1">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={() => skipTime(-5)}
            >
              <SkipBack className="w-4 h-4" />
            </Button>
            <Button
              variant="default"
              size="icon"
              className="w-12 h-12"
              onClick={togglePlay}
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={() => skipTime(5)}
            >
              <SkipForward className="w-4 h-4" />
            </Button>
          </div>

          <div className="flex items-center gap-3 flex-1 max-w-xs">
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleMute}
            >
              {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </Button>
            <div 
              className="flex-1 h-2 bg-muted rounded-full overflow-hidden cursor-pointer relative group"
              onClick={handleVolumeClick}
            >
              <div
                className="absolute top-0 bottom-0 left-0 bg-primary"
                style={{ width: `${(isMuted ? 0 : volume) * 100}%` }}
              />
              <div
                className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-primary rounded-full shadow-lg opacity-0 group-hover:opacity-100 transition-opacity"
                style={{ left: `${(isMuted ? 0 : volume) * 100}%`, marginLeft: '-6px' }}
              />
            </div>
          </div>

          <Button
            variant="outline"
            size="icon"
            onClick={toggleFullscreen}
          >
            <Maximize className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
