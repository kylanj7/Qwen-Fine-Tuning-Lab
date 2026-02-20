"""Training endpoints."""
import os
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..db.models import TrainingRun, RunStatus
from ..services.training_service import TrainingService

router = APIRouter()

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TrainingStartRequest(BaseModel):
    """Request to start a training run."""
    model_config_name: str
    dataset_config_name: str
    training_config_name: str
    parameter_overrides: Optional[dict] = {}


class TrainingStatusResponse(BaseModel):
    """Training run status response."""
    id: int
    run_name: str
    status: str
    current_step: int
    total_steps: int
    current_epoch: float
    total_epochs: int
    current_loss: Optional[float]
    best_loss: Optional[float]
    wandb_run_url: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class TrainingListResponse(BaseModel):
    """List of training runs."""
    runs: List[TrainingStatusResponse]
    total: int


@router.post("/start", response_model=TrainingStatusResponse)
async def start_training(
    request: TrainingStartRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start a new training run."""
    # Create training run record
    service = TrainingService(db)

    try:
        run = service.create_run(
            model_config_name=request.model_config_name,
            dataset_config_name=request.dataset_config_name,
            training_config_name=request.training_config_name,
            parameter_overrides=request.parameter_overrides
        )

        # Start training in background
        background_tasks.add_task(service.run_training, run.id)

        return run

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{run_id}", response_model=TrainingStatusResponse)
async def get_training_status(run_id: int, db: Session = Depends(get_db)):
    """Get training run status."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()

    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    return run


@router.get("/list", response_model=TrainingListResponse)
async def list_training_runs(
    skip: int = 0,
    limit: int = 20,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List training runs with pagination."""
    query = db.query(TrainingRun)

    if status:
        query = query.filter(TrainingRun.status == status)

    total = query.count()
    runs = query.order_by(TrainingRun.created_at.desc()).offset(skip).limit(limit).all()

    return TrainingListResponse(runs=runs, total=total)


@router.post("/cancel/{run_id}")
async def cancel_training(run_id: int, db: Session = Depends(get_db)):
    """Cancel a training run."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()

    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    if run.status not in [RunStatus.PENDING, RunStatus.RUNNING]:
        raise HTTPException(status_code=400, detail="Can only cancel pending or running jobs")

    service = TrainingService(db)
    service.cancel_run(run_id)

    return {"status": "cancelled", "run_id": run_id}


@router.get("/logs/{run_id}")
async def get_training_logs(run_id: int, lines: int = 100, db: Session = Depends(get_db)):
    """Get recent training logs."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()

    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    service = TrainingService(db)
    logs = service.get_logs(run_id, lines)

    return {"run_id": run_id, "logs": logs}
