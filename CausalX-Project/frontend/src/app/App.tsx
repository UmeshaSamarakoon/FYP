import { useMemo, useState } from 'react';
import { VideoUpload } from '@/app/components/VideoUpload';
import { VideoAnalysis } from '@/app/components/VideoAnalysis';
import { ResearchDescription } from '@/app/components/ResearchDescription';
import { Logo } from '@/app/components/Logo';
import { Button } from '@/app/components/ui/button';
import { ArrowLeft } from 'lucide-react';
import { analyzeVideo, FrameResult } from '@/app/lib/api';

type BreachSegment = { start: number; end: number; score: number };

interface AnalysisResult {
  result: 'REAL' | 'FAKE';
  confidence: number;
  confidenceLabel: string;
  previewUrl?: string | null;
  causalBreachScore?: number;
  breachSegments: BreachSegment[];
  frames: FrameResult[];
}

const PROB_THRESHOLD = 0.6;

function buildSegments(
  frames: FrameResult[],
  apiSegments?: { start: number; end: number; score?: number }[],
  threshold = PROB_THRESHOLD,
  maxGap = 0.5,
): BreachSegment[] {
  if (apiSegments?.length) {
    return apiSegments.map((segment) => ({
      start: segment.start,
      end: segment.end,
      score: segment.score ?? 0,
    }));
  }

  const suspicious = frames
    .filter((f) => (f.fake_prob ?? 0) >= threshold)
    .sort((a, b) => a.timestamp - b.timestamp);

  const segments: BreachSegment[] = [];
  let current: BreachSegment | null = null;
  const frameBreachScore = (f: FrameResult) => f.causal_breach_score ?? f.av_mismatch ?? f.fake_prob;

  for (const f of suspicious) {
    if (!current) {
      current = { start: f.timestamp, end: f.timestamp, score: frameBreachScore(f) };
      continue;
    }

    const gap = f.timestamp - current.end;
    if (gap <= maxGap) {
      current.end = f.timestamp;
      current.score = Math.max(current.score, frameBreachScore(f));
    } else {
      segments.push(current);
      current = { start: f.timestamp, end: f.timestamp, score: frameBreachScore(f) };
    }
  }

  if (current) segments.push(current);

  return segments;
}

export default function App() {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async (file: File) => {
    setIsAnalyzing(true);
    setError(null);
    setAnalysisResult(null);

    try {
      const res = await analyzeVideo(file);
      const label = typeof res.video_fake === "string"
        ? res.video_fake.toUpperCase()
        : res.video_fake ? "FAKE" : "REAL";
      const resultLabel: 'REAL' | 'FAKE' = label === "FAKE" ? "FAKE" : "REAL";
      const fakeLikelihood = Math.max(0, Math.min(1, Number(res.fake_confidence ?? 0)));
      const confidence = resultLabel === "FAKE"
        ? fakeLikelihood * 100
        : (1 - fakeLikelihood) * 100;
      const confidenceLabel = resultLabel === "FAKE" ? "Fake Likelihood" : "Real Confidence";

      const frames = res.frames ?? [];
      const segments = buildSegments(frames, res.causal_segments);

      setUploadedFile(file);
      setAnalysisResult({
        result: resultLabel,
        confidence,
        confidenceLabel,
        previewUrl: res.preview_url ?? null,
        causalBreachScore: resultLabel === "FAKE" ? res.causal_breach_score : undefined,
        breachSegments: resultLabel === "FAKE" ? segments : [],
        frames,
      });
    } catch (err: any) {
      setError(err?.message || "Analysis failed");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setUploadedFile(null);
    setAnalysisResult(null);
    setError(null);
  };

  const breachCount = useMemo(() => analysisResult?.breachSegments.length ?? 0, [analysisResult]);

  if (isAnalyzing) {
    return (
      <div className="min-h-screen bg-background">
        <div className="absolute top-6 left-6">
          <Logo />
        </div>
        <div className="flex items-center justify-center min-h-screen">
          <div className="text-center space-y-6">
            <div className="w-20 h-20 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
            <div>
              <h2 className="text-3xl font-bold mb-2">Analyzing Video</h2>
              <p className="text-muted-foreground text-lg">
                Running Causal Fusion Network...
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (uploadedFile && analysisResult) {
    return (
      <div className="min-h-screen bg-background p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                variant="outline"
                onClick={handleReset}
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Upload New Video
              </Button>
              <Logo />
            </div>
            <div className="text-sm text-muted-foreground">
              Breach segments: {breachCount}
            </div>
          </div>
          
          <div className="flex items-center justify-center">
            <VideoAnalysis
              videoFile={uploadedFile}
              previewUrl={analysisResult.previewUrl}
              result={analysisResult.result}
              confidence={analysisResult.confidence}
              confidenceLabel={analysisResult.confidenceLabel}
              breachSegments={analysisResult.breachSegments}
              frames={analysisResult.frames}
              causalBreachScore={analysisResult.causalBreachScore}
              probThreshold={PROB_THRESHOLD}
            />
          </div>

          <div className="pt-6">
            <ResearchDescription />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <VideoUpload onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} error={error} />
    </div>
  );
}
