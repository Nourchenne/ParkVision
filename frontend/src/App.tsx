import { createContext, useContext } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { Layout } from './layouts/Layout';
import { OverviewPage } from './pages/OverviewPage';
import { VideoPage } from './pages/VideoPage';
import { ImagePage } from './pages/ImagePage';
import { ModelPage } from './pages/ModelPage';
import { DiagnosticsPage } from './pages/DiagnosticsPage';
import { useMonitoring } from './hooks/useMonitoring';

type MonitoringState = ReturnType<typeof useMonitoring>;
const MonitoringContext = createContext<MonitoringState | null>(null);

export function useApp() {
  const value = useContext(MonitoringContext);
  if (!value) throw new Error('Monitoring provider is missing');
  return value;
}

function Content() {
  const monitoring = useMonitoring();
  return (
    <MonitoringContext.Provider value={monitoring}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<OverviewPage />} />
            <Route path="video" element={<VideoPage />} />
            <Route path="image" element={<ImagePage />} />
            <Route path="model" element={<ModelPage />} />
            <Route path="diagnostics" element={<DiagnosticsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </MonitoringContext.Provider>
  );
}

export default Content;
