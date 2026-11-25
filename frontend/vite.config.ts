import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

const target = 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: Object.fromEntries(
      ['/health', '/ready', '/model-info', '/predict', '/video'].map((route) => [
        route,
        { target, changeOrigin: true, ws: route === '/video' },
      ]),
    ),
  },
});
