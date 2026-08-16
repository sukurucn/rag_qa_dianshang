import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/query": "http://127.0.0.1:8000",
      "/sessions": "http://127.0.0.1:8000",
      "/runtime": "http://127.0.0.1:8000",
      "/admin": "http://127.0.0.1:8001"
    }
  }
});
