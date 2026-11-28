// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ModelInfo } from '../types/api';
import { ModelPage } from './ModelPage';

type AppState = { model: ModelInfo | null; ready: boolean | null; health: boolean | null };
let state: AppState;
vi.mock('../App', () => ({ useApp: () => state }));

const champion: ModelInfo = {
  name: 'ParkVision-Detector', alias: 'champion', version: '2', architecture: 'yolov8s',
  source_run_id: 'run-abc123', model_uri: 'models:/ParkVision-Detector@champion', status: 'loaded',
};

afterEach(cleanup);

describe('ModelPage', () => {
  it('shows the exact champion version and run when the model is loaded', () => {
    state = { model: champion, ready: true, health: true };
    render(<ModelPage />);
    expect(screen.getByRole('heading', { name: 'ParkVision-Detector' })).toBeTruthy();
    expect(screen.getByText('Ready for inference')).toBeTruthy();
    expect(screen.getByText('run-abc123')).toBeTruthy();
    expect(screen.getByText('models:/ParkVision-Detector@champion')).toBeTruthy();
    expect(screen.getByText('Online')).toBeTruthy();
    expect(screen.getByText('Ready')).toBeTruthy();
  });

  it('reports metadata as unavailable when the API is offline', () => {
    state = { model: null, ready: false, health: false };
    render(<ModelPage />);
    expect(screen.getByText('Model details unavailable')).toBeTruthy();
    expect(screen.getByText('Metadata unavailable')).toBeTruthy();
    expect(screen.getByText('Offline')).toBeTruthy();
    expect(screen.getAllByText('Unavailable').length).toBeGreaterThan(0);
  });

  it('shows a loading state while the first health check is pending', () => {
    state = { model: null, ready: null, health: null };
    const { container } = render(<ModelPage />);
    expect(container.querySelector('.skeleton')).toBeTruthy();
    expect(screen.getAllByText('Checking').length).toBe(2);
  });
});
