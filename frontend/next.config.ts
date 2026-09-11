import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  env: {
    // Where the FastAPI backend listens. Overridable so the demo can point elsewhere.
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
  },
};

export default config;
