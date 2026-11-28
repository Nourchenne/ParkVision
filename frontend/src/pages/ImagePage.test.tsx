// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { api, ApiError } from '../services/api';
import type { Prediction } from '../types/api';
import { ImagePage } from './ImagePage';

const prediction: Prediction = {
  model: { name: 'ParkVision-Detector', version: '2', architecture: 'yolov8s' },
  counts: { empty: 7, occupied: 93, total: 100 },
  detections: [
    { class_id: 0, class_name: 'space-empty', confidence: 0.912, bbox: [10.2, 20.4, 60.6, 90.1] },
    { class_id: 1, class_name: 'space-occupied', confidence: 0.55, bbox: [100, 120, 160, 190] },
  ],
  inference_time_ms: 42.37,
};

const photo = () => new File(['x'], 'lot.jpg', { type: 'image/jpeg' });
const chooseFile = (file: File) =>
  fireEvent.change(screen.getByLabelText('Choose image file'), { target: { files: [file] } });

beforeEach(() => {
  vi.stubGlobal('URL', Object.assign(URL, {
    createObjectURL: vi.fn(() => 'blob:fake'), revokeObjectURL: vi.fn(),
  }));
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('ImagePage', () => {
  it('starts empty with the analyze button disabled', () => {
    render(<ImagePage />);
    expect(screen.getByText('No image selected')).toBeTruthy();
    expect((screen.getByRole('button', { name: /Analyze image/ }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText('No detections yet')).toBeTruthy();
  });

  it('previews the chosen file and enables the analyze button', () => {
    render(<ImagePage />);
    chooseFile(photo());
    expect(screen.getByAltText('Selected parking lot')).toBeTruthy();
    expect(screen.getByText('lot.jpg')).toBeTruthy();
    expect((screen.getByRole('button', { name: /Analyze image/ }) as HTMLButtonElement).disabled).toBe(false);
  });

  it('shows counts, latency and detections after a successful analysis', async () => {
    const predict = vi.spyOn(api, 'predict').mockResolvedValue(prediction);
    vi.spyOn(api, 'predictVisualize').mockResolvedValue('blob:annotated');
    render(<ImagePage />);
    const file = photo();
    chooseFile(file);
    fireEvent.click(screen.getByRole('button', { name: /Analyze image/ }));
    await waitFor(() => expect(screen.getByText('93')).toBeTruthy());
    expect(predict).toHaveBeenCalledWith(file);
    expect(screen.getByText('7')).toBeTruthy();
    expect(screen.getByText('100')).toBeTruthy();
    expect(screen.getByText('42.4 ms')).toBeTruthy();
    expect(screen.getByAltText('Annotated parking lot with detected spaces')).toBeTruthy();
    expect(screen.getByText('91.2%')).toBeTruthy();
    expect(screen.getByText('space-occupied')).toBeTruthy();
  });

  it('shows the API error message when the analysis fails', async () => {
    vi.spyOn(api, 'predict').mockRejectedValue(new ApiError('Image is too large.', 413, 'too_large'));
    vi.spyOn(api, 'predictVisualize').mockResolvedValue('blob:annotated');
    render(<ImagePage />);
    chooseFile(photo());
    fireEvent.click(screen.getByRole('button', { name: /Analyze image/ }));
    await waitFor(() => expect(screen.getByText('Image is too large.')).toBeTruthy());
    expect(screen.getByText('No detections yet')).toBeTruthy();
  });

  it('falls back to a generic message for unexpected failures', async () => {
    vi.spyOn(api, 'predict').mockRejectedValue(new Error('boom'));
    vi.spyOn(api, 'predictVisualize').mockResolvedValue('blob:annotated');
    render(<ImagePage />);
    chooseFile(photo());
    fireEvent.click(screen.getByRole('button', { name: /Analyze image/ }));
    await waitFor(() => expect(screen.getByText('Image analysis failed.')).toBeTruthy());
  });
});
