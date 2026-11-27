import { useEffect, useRef, useState } from 'react';
import { ArrowRight, CircleAlert, Clock3, CloudUpload, ImageOff, ParkingSquare,
  ScanLine, Square } from 'lucide-react';
import { MetricCard, PageIntro, Panel } from '../components/UI';
import { api, ApiError } from '../services/api';
import type { Prediction } from '../types/api';

export function ImagePage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ prediction: Prediction; annotatedUrl: string } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (result) URL.revokeObjectURL(result.annotatedUrl);
  }, [previewUrl, result]);

  const selectFile = (value?: File) => {
    if (!value) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (result) { URL.revokeObjectURL(result.annotatedUrl); setResult(null); }
    setError(null);
    setFile(value);
    setPreviewUrl(URL.createObjectURL(value));
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const [prediction, annotatedUrl] = await Promise.all([api.predict(file), api.predictVisualize(file)]);
      if (result) URL.revokeObjectURL(result.annotatedUrl);
      setResult({ prediction, annotatedUrl });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Image analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  const counts = result?.prediction.counts;
  const detections = result?.prediction.detections ?? [];

  return <>
    <PageIntro eyebrow="IMAGE ANALYSIS" title="Analyze a single image"
      text="Upload a parking lot photo to run detection once, outside of a video session. Useful for a quick check, or for testing a shot from a gallery or a webcam." />
    <div className="image-page-grid">
      <div className="image-main">
        <Panel title="Source image" eyebrow="ORIGINAL">
          <div className="image-stage">{previewUrl ?
            <img src={previewUrl} alt="Selected parking lot" /> :
            <div className="image-empty"><ImageOff size={32} /><strong>No image selected</strong>
              <p>Choose a photo to preview it here.</p></div>}</div>
        </Panel>
        <Panel title="Résultats" eyebrow="LIVE METRICS" tone="red">
          <div className="metric-grid">
            <MetricCard label="Empty spaces" value={counts?.empty ?? '—'} detail="This image" tone="green" icon={<ParkingSquare size={19} />} />
            <MetricCard label="Occupied spaces" value={counts?.occupied ?? '—'} detail="This image" tone="red" icon={<Square size={19} />} />
            <MetricCard label="Total detected" value={counts?.total ?? '—'} detail="Empty + occupied" tone="amber" icon={<ScanLine size={19} />} />
            <MetricCard label="Inference latency" value={result ? `${result.prediction.inference_time_ms.toFixed(1)} ms` : '—'}
              detail="This image" icon={<Clock3 size={19} />} />
          </div>
        </Panel>
        <Panel title="Annotated result" eyebrow="DETECTION OUTPUT">
          <div className="image-stage">{result ?
            <img src={result.annotatedUrl} alt="Annotated parking lot with detected spaces" /> :
            <div className="image-empty"><ScanLine size={32} /><p>Run an analysis to see the annotated image here.</p></div>}</div>
        </Panel>
        <Panel title="Detected spaces" eyebrow="DETECTIONS" tone="blue">
          {detections.length > 0 ? <table className="detections-table">
            <thead><tr><th>Class</th><th>Confidence</th><th>Box (x1, y1, x2, y2)</th></tr></thead>
            <tbody>{detections.map((detection, index) => <tr key={index}>
              <td><span className={`tag ${detection.class_name === 'empty' ? 'tag-empty' : 'tag-occupied'}`}>{detection.class_name}</span></td>
              <td>{(detection.confidence * 100).toFixed(1)}%</td>
              <td>{detection.bbox.map((value) => value.toFixed(0)).join(', ')}</td>
            </tr>)}</tbody>
          </table> : <div className="empty-panel"><ScanLine size={30} /><strong>No detections yet</strong>
            <p>Analyze an image to list every detected parking space here.</p></div>}
        </Panel>
      </div>
      <div className="image-side">
        <Panel title="Upload image" eyebrow="IMAGE SOURCE">
          <div className={`drop-zone ${dragging ? 'dragging' : ''}`} role="button" tabIndex={0}
            onClick={() => fileInput.current?.click()} onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') fileInput.current?.click();
            }} onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)} onDrop={(event) => {
              event.preventDefault(); setDragging(false); selectFile(event.dataTransfer.files[0]);
            }}>
            <span className="drop-icon"><CloudUpload size={25} /></span>
            <strong>{file?.name ?? 'Drop a JPG or PNG here'}</strong>
            <span>{file ? `${(file.size / 1048576).toFixed(1)} MB · Click to change` : 'or click to browse your files'}</span>
            <input ref={fileInput} type="file" accept="image/jpeg,image/png,.jpg,.jpeg,.png" aria-label="Choose image file"
              onChange={(event) => selectFile(event.target.files?.[0])} hidden />
          </div>
          <button className="button button-primary button-wide" disabled={!file || loading} onClick={() => void analyze()}>
            {loading ? 'Analyzing…' : 'Analyze image'} <ArrowRight size={18} />
          </button>
          <p className="form-hint">A gallery picker and webcam capture can reuse this same upload flow.</p>
          {error && <p className="inline-error"><CircleAlert size={17} />{error}</p>}
        </Panel>
      </div>
    </div>
  </>;
}
