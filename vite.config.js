import { defineConfig } from 'vite';
import { readFileSync } from 'node:fs';
import { buildData } from './web/data.js';

const dataPath = '/scaling-data.json';

export default defineConfig({
  plugins: [{
    name: 'scaling-data',
    buildStart() {
      this.addWatchFile(new URL('./results.csv', import.meta.url).pathname);
    },
    configureServer(server) {
      server.middlewares.use(dataPath, (_request, response, next) => {
        try {
          response.setHeader('Content-Type', 'application/json');
          response.end(JSON.stringify(loadData()));
        } catch (error) {
          next(error);
        }
      });
    },
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'scaling-data.json',
        source: JSON.stringify(loadData()),
      });
    },
  }],
});

function loadData() {
  return buildData(readFileSync(new URL('./results.csv', import.meta.url), 'utf8'));
}
