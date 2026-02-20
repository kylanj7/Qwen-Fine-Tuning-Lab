"""FastAPI main application entry point."""
import os
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add parent directory to path for importing existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .db.database import init_db
from .routers import training, evaluation, models, inference, configs, websocket, papers
from .routers.websocket import broadcast_consumer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    init_db()
    print("Database initialized")

    # Start the broadcast consumer task
    consumer_task = asyncio.create_task(broadcast_consumer())
    print("Broadcast consumer started")

    yield

    # Shutdown
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    print("Application shutting down")


app = FastAPI(
    title="Qwen Fine Tune Test Suite",
    description="API for training, evaluating, and testing Qwen language models",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(configs.router, prefix="/api/configs", tags=["configs"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["evaluation"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(papers.router, prefix="/api/papers", tags=["papers"])
app.include_router(websocket.router, tags=["websocket"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Qwen Fine Tune Test Suite API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
