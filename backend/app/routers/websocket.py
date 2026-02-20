"""WebSocket handlers for real-time updates."""
import os
import sys
import json
import queue
import asyncio
from typing import Dict, Set
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session

from ..db.database import get_db, SessionLocal
from ..db.models import TrainingRun, EvaluationRun, ChatSession
from ..broadcast_queue import get_broadcast_queue, BroadcastType

# Add parent directory to path for importing existing modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, BASE_DIR)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        # Connections organized by type and ID
        self.training_connections: Dict[int, Set[WebSocket]] = {}
        self.evaluation_connections: Dict[int, Set[WebSocket]] = {}
        self.inference_connections: Dict[int, Set[WebSocket]] = {}

    async def connect_training(self, websocket: WebSocket, run_id: int):
        await websocket.accept()
        if run_id not in self.training_connections:
            self.training_connections[run_id] = set()
        self.training_connections[run_id].add(websocket)

    async def connect_evaluation(self, websocket: WebSocket, eval_id: int):
        await websocket.accept()
        if eval_id not in self.evaluation_connections:
            self.evaluation_connections[eval_id] = set()
        self.evaluation_connections[eval_id].add(websocket)
        print(f"[WebSocket] Client connected for evaluation {eval_id}. Total clients: {len(self.evaluation_connections[eval_id])}", flush=True)

    async def connect_inference(self, websocket: WebSocket, session_id: int):
        await websocket.accept()
        if session_id not in self.inference_connections:
            self.inference_connections[session_id] = set()
        self.inference_connections[session_id].add(websocket)

    def disconnect_training(self, websocket: WebSocket, run_id: int):
        if run_id in self.training_connections:
            self.training_connections[run_id].discard(websocket)

    def disconnect_evaluation(self, websocket: WebSocket, eval_id: int):
        if eval_id in self.evaluation_connections:
            self.evaluation_connections[eval_id].discard(websocket)
            print(f"[WebSocket] Client disconnected from evaluation {eval_id}. Remaining clients: {len(self.evaluation_connections[eval_id])}", flush=True)

    def disconnect_inference(self, websocket: WebSocket, session_id: int):
        if session_id in self.inference_connections:
            self.inference_connections[session_id].discard(websocket)

    async def broadcast_training(self, run_id: int, message: dict):
        if run_id in self.training_connections:
            for connection in list(self.training_connections[run_id]):
                try:
                    await connection.send_json(message)
                except Exception:
                    self.training_connections[run_id].discard(connection)

    async def broadcast_evaluation(self, eval_id: int, message: dict):
        if eval_id in self.evaluation_connections:
            connections = list(self.evaluation_connections[eval_id])
            print(f"[WebSocket] Sending to {len(connections)} clients for eval {eval_id}: {message.get('type', 'unknown')}", flush=True)
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"[WebSocket] Send error: {e}", flush=True)
                    self.evaluation_connections[eval_id].discard(connection)
        else:
            print(f"[WebSocket] No clients connected for eval {eval_id}", flush=True)

    async def send_inference(self, session_id: int, message: dict):
        if session_id in self.inference_connections:
            for connection in list(self.inference_connections[session_id]):
                try:
                    await connection.send_json(message)
                except Exception:
                    self.inference_connections[session_id].discard(connection)


manager = ConnectionManager()


@router.websocket("/ws/training/{run_id}")
async def training_websocket(websocket: WebSocket, run_id: int):
    """WebSocket for streaming training progress."""
    await manager.connect_training(websocket, run_id)

    try:
        # Send initial status
        db = SessionLocal()
        try:
            run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run:
                await websocket.send_json({
                    "type": "status",
                    "data": {
                        "status": run.status,
                        "current_step": run.current_step,
                        "total_steps": run.total_steps,
                        "current_epoch": run.current_epoch,
                        "total_epochs": run.total_epochs,
                        "current_loss": run.current_loss,
                        "best_loss": run.best_loss
                    }
                })
        finally:
            db.close()

        # Keep connection alive and handle messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle ping/pong for keepalive
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        manager.disconnect_training(websocket, run_id)


@router.websocket("/ws/evaluation/{eval_id}")
async def evaluation_websocket(websocket: WebSocket, eval_id: int):
    """WebSocket for streaming evaluation progress."""
    await manager.connect_evaluation(websocket, eval_id)

    try:
        # Send initial status
        db = SessionLocal()
        try:
            evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == eval_id).first()
            if evaluation:
                await websocket.send_json({
                    "type": "status",
                    "data": {
                        "status": evaluation.status,
                        "current_sample": evaluation.current_sample,
                        "total_samples": evaluation.total_samples,
                        "factual_accuracy": evaluation.factual_accuracy,
                        "completeness": evaluation.completeness,
                        "technical_precision": evaluation.technical_precision,
                        "overall_score": evaluation.overall_score
                    }
                })
        finally:
            db.close()

        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        manager.disconnect_evaluation(websocket, eval_id)


@router.websocket("/ws/inference/{session_id}")
async def inference_websocket(websocket: WebSocket, session_id: int):
    """WebSocket for streaming chat inference."""
    await manager.connect_inference(websocket, session_id)

    try:
        db = SessionLocal()

        while True:
            # Receive chat message
            data = await websocket.receive_json()

            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if data.get("type") == "message":
                message = data.get("content", "")
                model_path = data.get("model_path")
                temperature = data.get("temperature", 0.7)
                top_p = data.get("top_p", 0.9)
                top_k = data.get("top_k", 40)
                max_tokens = data.get("max_tokens", 2048)
                system_prompt = data.get("system_prompt", "")

                # Get session for chat history
                session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
                if not session:
                    await websocket.send_json({"type": "error", "message": "Session not found"})
                    continue

                # Add user message to history
                messages = session.messages or []
                messages.append({
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Generate response with streaming
                try:
                    from llama_cpp import Llama

                    # Load model (with caching in production)
                    llm = Llama(
                        model_path=model_path,
                        n_ctx=4096,
                        n_gpu_layers=-1,  # Use all GPU layers
                        verbose=False
                    )

                    # Format prompt
                    prompt = format_qwen_prompt(messages, system_prompt)

                    # Send start signal
                    await websocket.send_json({"type": "start"})

                    # Stream response
                    full_response = ""
                    token_count = 0
                    start_time = datetime.utcnow()

                    for output in llm(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        stream=True
                    ):
                        token = output["choices"][0]["text"]
                        full_response += token
                        token_count += 1

                        await websocket.send_json({
                            "type": "token",
                            "content": token,
                            "token_count": token_count
                        })

                    # Calculate metrics
                    elapsed = (datetime.utcnow() - start_time).total_seconds()
                    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0

                    # Add assistant message to history
                    messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.utcnow().isoformat(),
                        "tokens": token_count,
                        "tokens_per_sec": round(tokens_per_sec, 2)
                    })

                    # Update session
                    session.messages = messages
                    db.commit()

                    # Send completion signal
                    await websocket.send_json({
                        "type": "complete",
                        "tokens": token_count,
                        "tokens_per_sec": round(tokens_per_sec, 2),
                        "elapsed_sec": round(elapsed, 2)
                    })

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })

    except WebSocketDisconnect:
        manager.disconnect_inference(websocket, session_id)
    finally:
        db.close()


def format_qwen_prompt(messages: list, system_prompt: str = "") -> str:
    """Format messages into Qwen chat format."""
    prompt = ""

    if system_prompt:
        prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"
    return prompt


# Export manager for use by services
def get_connection_manager() -> ConnectionManager:
    return manager


async def broadcast_consumer():
    """Async consumer that reads from queue and broadcasts to WebSocket clients.

    This runs as a background task and bridges sync code (evaluation_service)
    to async WebSocket broadcasts.
    """
    print("[BroadcastConsumer] Started", flush=True)
    bq = get_broadcast_queue()
    while True:
        try:
            # Non-blocking get from the queue
            msg = bq.get_nowait()
            print(f"[BroadcastConsumer] Got message: {msg.broadcast_type} for entity {msg.entity_id}", flush=True)
            if msg.broadcast_type == BroadcastType.EVALUATION:
                num_clients = len(manager.evaluation_connections.get(msg.entity_id, set()))
                print(f"[BroadcastConsumer] Broadcasting to {num_clients} evaluation clients for eval {msg.entity_id}", flush=True)
                await manager.broadcast_evaluation(msg.entity_id, msg.message)
            elif msg.broadcast_type == BroadcastType.TRAINING:
                await manager.broadcast_training(msg.entity_id, msg.message)
        except queue.Empty:
            # No messages, sleep briefly to avoid busy-waiting
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            # Graceful shutdown
            print("[BroadcastConsumer] Shutting down", flush=True)
            break
        except Exception as e:
            # Log but don't crash the consumer
            print(f"[BroadcastConsumer] Error: {e}", flush=True)
