# Qwen Fine Tune Test Suite - Frontend

React + TypeScript frontend for the Qwen Fine Tune Test Suite.

## Quick Start

### Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`.

### Production (Docker)

```bash
# From the project root
docker compose up frontend backend
```

The frontend will be available at `http://localhost:3000`.

## Features

### Training Dashboard
- Select model, dataset, and training configurations
- Override training parameters (learning rate, epochs, batch size)
- Real-time training progress via WebSocket
- Live log viewer
- WandB integration link

### Evaluation Dashboard
- Select GGUF model and test dataset
- Configure sample count and judge model
- Real-time evaluation progress
- 3-Dimension radar chart (Factual Accuracy, Completeness, Technical Precision)
- Expandable per-question results with RAG sources

### Model Management
- List LoRA adapters from training runs
- List GGUF models for inference
- Convert adapters to GGUF with quantization options
- Delete unused models

### Chat Interface
- Select GGUF model for inference
- Configurable system prompt
- Generation parameter sliders (temperature, top_p, top_k, max_tokens)
- Streaming chat with real-time token metrics

### History & Comparison
- Training run history with loss metrics
- Evaluation history with multi-model comparison
- Radar chart overlay for comparing evaluations
- Export to JSON

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Zustand** - State management
- **Recharts** - Charts (radar charts)
- **CSS Modules** - Scoped styling

## Project Structure

```
src/
├── api/              # API client
│   └── client.ts     # Typed API functions
├── components/
│   ├── common/       # Layout, Card, Button, etc.
│   └── charts/       # RadarChart
├── hooks/
│   └── useWebSocket.ts  # WebSocket hooks
├── pages/
│   ├── Training.tsx
│   ├── Evaluation.tsx
│   ├── Models.tsx
│   ├── Chat.tsx
│   └── History.tsx
├── store/            # Zustand stores
│   ├── configStore.ts
│   ├── trainingStore.ts
│   ├── evaluationStore.ts
│   └── modelStore.ts
└── styles/
    ├── variables.css # Design tokens
    └── global.css    # Global styles
```

## Design System

Scientific/lab aesthetic with professional blues:

```css
--primary-900: #0d1b2a;     /* Deep navy (background) */
--primary-800: #1b263b;     /* Dark blue (cards) */
--accent-cyan: #00d4ff;     /* Highlights, progress */
--accent-green: #00ff88;    /* Success states */

/* Score Colors */
--score-factual: #3b82f6;   /* Blue */
--score-complete: #22c55e;  /* Green */
--score-precision: #f59e0b; /* Amber */
```

## Backend API

The frontend expects the backend API at `/api`. In development, Vite proxies requests to `http://localhost:8000`.

Key endpoints:
- `GET /api/configs/all` - List all configurations
- `POST /api/training/start` - Start training
- `WS /ws/training/{id}` - Training progress stream
- `POST /api/evaluation/start` - Start evaluation
- `WS /ws/evaluation/{id}` - Evaluation progress stream
- `GET /api/models/adapters` - List LoRA adapters
- `GET /api/models/gguf` - List GGUF models
- `WS /ws/inference/{id}` - Chat streaming
