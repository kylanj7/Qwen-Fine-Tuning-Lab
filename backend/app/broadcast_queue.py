"""Thread-safe queue for sync-to-async broadcast communication."""
import queue
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class BroadcastType(str, Enum):
    EVALUATION = "evaluation"
    TRAINING = "training"


@dataclass
class BroadcastMessage:
    broadcast_type: BroadcastType
    entity_id: int
    message: Dict[str, Any]


_broadcast_queue: "queue.Queue[BroadcastMessage]" = queue.Queue()


def enqueue_broadcast(broadcast_type: BroadcastType, entity_id: int, message: Dict[str, Any]):
    """Add a message to the broadcast queue (thread-safe, can be called from sync code)."""
    _broadcast_queue.put(BroadcastMessage(broadcast_type, entity_id, message))
    print(f"[BroadcastQueue] Enqueued {message.get('type', 'unknown')} for {broadcast_type.value} {entity_id}. Queue size: {_broadcast_queue.qsize()}", flush=True)


def get_broadcast_queue() -> "queue.Queue[BroadcastMessage]":
    """Get the broadcast queue instance."""
    return _broadcast_queue
