import { useState, useCallback } from 'react';
import { Upload, Video, AlertTriangle } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Logo } from '@/app/components/Logo';

interface VideoUploadProps {
  onAnalyze: (file: File) => void;
  isAnalyzing?: boolean;
  error?: string | null;
}

export function VideoUpload({ onAnalyze, isAnalyzing = false, error }: VideoUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/')) {
        setSelectedFile(file);
      }
    }
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  return (
    <div className="min-h-screen relative">
      {/* Logo */}
      <div className="absolute top-6 left-6">
        <Logo />
      </div>
      
      <div className="flex items-center justify-center min-h-[80vh] pt-20">
        <div className="max-w-2xl w-full space-y-6 px-6">
          <div className="text-center space-y-2">
            <h1 className="text-5xl font-bold">CausalX</h1>
            <p className="text-xl text-muted-foreground">
              Deepfake Detection System
            </p>
          </div>

          <div
            className={`relative border-2 border-dashed rounded-lg p-16 transition-colors ${
              dragActive
                ? 'border-primary bg-primary/5'
                : 'border-muted-foreground/25 hover:border-muted-foreground/50'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              id="video-upload"
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              accept="video/*"
              onChange={handleChange}
              disabled={isAnalyzing}
            />
            
            <div className="flex flex-col items-center gap-6 text-center">
              <div className="p-6 bg-primary/10 rounded-full">
                <Upload className="w-12 h-12 text-primary" />
              </div>
              <div>
                <p className="text-2xl font-medium">
                  Upload Video
                </p>
                <p className="text-muted-foreground mt-2">
                  Drag and drop or click to browse
                </p>
              </div>
            </div>
          </div>

          {selectedFile && (
            <div className="space-y-4">
              <div className="flex items-center gap-4 p-4 bg-muted rounded-lg">
                <Video className="w-8 h-8 text-primary shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{selectedFile.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
              </div>

              <Button
                onClick={() => onAnalyze(selectedFile)}
                className="w-full"
                size="lg"
                disabled={isAnalyzing}
              >
                {isAnalyzing ? "Analyzing..." : "Analyze Video"}
              </Button>
            </div>
          )}

          {error && (
            <div className="flex items-center gap-3 p-3 rounded-lg bg-red-500/10 text-red-500 text-sm">
              <AlertTriangle className="w-4 h-4" />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
