import { Card, CardContent, CardHeader, CardTitle } from '@/app/components/ui/card';

export function EvaluationMetrics() {
  const classificationMetrics = [
    { name: 'Accuracy', value: 94.2 },
    { name: 'Precision', value: 92.8 },
    { name: 'Recall', value: 95.1 },
    { name: 'F1-Score', value: 93.9 },
    { name: 'ROC-AUC', value: 96.2 },
  ];

  const explainabilityMetrics = [
    { name: 'Fidelity', value: 91 },
    { name: 'Faithfulness', value: 88 },
    { name: 'Robustness', value: 86 },
  ];

  const userMetrics = [
    { name: 'Trust', value: 4.3, max: 5 },
    { name: 'Intuitiveness', value: 4.5, max: 5 },
    { name: 'Usability', value: 4.6, max: 5 },
  ];

  return (
    <div className="grid gap-6 md:grid-cols-3">
      {/* Classification Performance */}
      <Card>
        <CardHeader>
          <CardTitle>Classification Performance</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {classificationMetrics.map((metric, idx) => (
            <div key={idx} className="flex items-center justify-between p-3 bg-muted rounded-lg">
              <span className="text-sm font-medium">{metric.name}</span>
              <span className="text-2xl font-bold text-primary">{metric.value}%</span>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Explainability Metrics */}
      <Card>
        <CardHeader>
          <CardTitle>Explainability Metrics</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {explainabilityMetrics.map((metric, idx) => (
            <div key={idx} className="flex items-center justify-between p-3 bg-muted rounded-lg">
              <span className="text-sm font-medium">{metric.name}</span>
              <span className="text-2xl font-bold text-primary">{metric.value}/100</span>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* User-Centric Evaluation */}
      <Card>
        <CardHeader>
          <CardTitle>User-Centric Evaluation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {userMetrics.map((metric, idx) => (
            <div key={idx} className="flex items-center justify-between p-3 bg-muted rounded-lg">
              <span className="text-sm font-medium">{metric.name}</span>
              <span className="text-2xl font-bold text-primary">{metric.value}/{metric.max}</span>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}