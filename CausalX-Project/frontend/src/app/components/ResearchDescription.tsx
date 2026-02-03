import { Card, CardContent, CardHeader, CardTitle } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';

export function ResearchDescription() {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Research Overview</h2>
      
      <div className="grid gap-6 md:grid-cols-2">
        {/* Research Objective */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Research Objective
              <Badge className="ml-auto">Core Mission</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground leading-relaxed">
              To develop an <span className="font-bold text-foreground">explainable-by-design deepfake detection system</span> that 
              achieves high classification accuracy while providing transparent, interpretable, and trustworthy visual explanations 
              through causal analysis.
            </p>
          </CardContent>
        </Card>

        {/* Primary Causal Factor */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Primary Causal Factor
              <Badge variant="destructive" className="ml-auto">Detection Method</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground leading-relaxed">
              <span className="font-bold text-foreground">Breakdown of Audio-Visual Speech Causality</span> - Detection focuses 
              on lip-sync temporal coherence as the primary indicator of manipulation. Natural speech exhibits strong temporal 
              correlation between lip movements and audio waveforms.
            </p>
          </CardContent>
        </Card>

        {/* Methodology: CFN */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Causal Flow Network (CFN)
              <Badge variant="outline" className="ml-auto">Component 1</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground leading-relaxed">
              Neural architecture designed to learn and model causal relationships between audio and visual modalities. 
              Outputs causal breach scores that directly indicate the probability of manipulation through multi-modal 
              feature extraction and temporal causal modeling.
            </p>
          </CardContent>
        </Card>

        {/* Methodology: CVI */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Causal Visualization Interface (CVI)
              <Badge variant="outline" className="ml-auto">Component 2</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground leading-relaxed">
              User interface layer that translates CFN outputs into intuitive visual explanations. Provides temporal 
              breach timelines and spatial bounding boxes for human-interpretable explanations without requiring 
              technical expertise.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Explainable-by-Design Philosophy */}
      <Card className="border-primary/50">
        <CardHeader>
          <CardTitle>Explainable-by-Design Philosophy</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <h4 className="font-semibold text-primary">Transparency</h4>
              <p className="text-sm text-muted-foreground">
                Every decision is backed by visual evidence showing where and when causal breaches occur
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold text-primary">Interpretability</h4>
              <p className="text-sm text-muted-foreground">
                Non-technical users can understand results through intuitive temporal and spatial visualizations
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold text-primary">Trustworthiness</h4>
              <p className="text-sm text-muted-foreground">
                Rigorous evaluation of both classification performance and explainability metrics
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Technical Pipeline */}
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle>Technical Pipeline</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <Badge variant="secondary">Video Input</Badge>
            <span className="text-muted-foreground">→</span>
            <Badge variant="secondary">Audio/Visual Extraction</Badge>
            <span className="text-muted-foreground">→</span>
            <Badge variant="secondary">CFN Processing</Badge>
            <span className="text-muted-foreground">→</span>
            <Badge variant="secondary">Causal Analysis</Badge>
            <span className="text-muted-foreground">→</span>
            <Badge variant="secondary">Visual Explanation</Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
