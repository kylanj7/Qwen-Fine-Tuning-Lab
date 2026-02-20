"""Model management endpoints."""
import os
import json
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..db.models import ConversionJob, RunStatus
from ..services.model_service import ModelService

router = APIRouter()

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
GGUF_DIR = os.path.join(BASE_DIR, "models", "gguf")


class LoRAAdapter(BaseModel):
    """LoRA adapter info."""
    name: str
    path: str
    base_model: Optional[str]
    dataset: Optional[str]
    created_at: Optional[datetime]
    size_mb: float


class GGUFModel(BaseModel):
    """GGUF model info."""
    name: str
    path: str
    filename: str
    size_mb: float
    quantization: Optional[str]
    created_at: Optional[datetime]


class ConversionRequest(BaseModel):
    """Request to convert a LoRA adapter to GGUF."""
    adapter_path: str
    adapter_name: str
    base_model: Optional[str] = None
    quantization_method: str = "q4_k_m"
    output_name: Optional[str] = None


class ConversionStatusResponse(BaseModel):
    """Conversion job status."""
    id: int
    adapter_name: str
    base_model: str
    quantization_method: str
    status: str
    current_stage: str
    progress: int
    output_path: Optional[str]
    output_size_bytes: Optional[int]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]

    class Config:
        from_attributes = True


@router.get("/adapters", response_model=List[LoRAAdapter])
async def list_adapters():
    """List all LoRA adapters."""
    adapters = []

    if not os.path.exists(OUTPUTS_DIR):
        return adapters

    for run_name in os.listdir(OUTPUTS_DIR):
        run_dir = os.path.join(OUTPUTS_DIR, run_name)
        adapter_path = os.path.join(run_dir, "final_adapter")

        if os.path.isdir(adapter_path):
            # Try to read metadata
            metadata_path = os.path.join(run_dir, "run_metadata.json")
            base_model = None
            dataset = None
            created_at = None

            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    base_model = metadata.get("base_model")
                    dataset = metadata.get("dataset")
                except Exception:
                    pass

            # Get directory size
            total_size = 0
            for dirpath, _, filenames in os.walk(adapter_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)

            # Get creation time
            try:
                created_at = datetime.fromtimestamp(os.path.getctime(adapter_path))
            except Exception:
                pass

            adapters.append(LoRAAdapter(
                name=run_name,
                path=adapter_path,
                base_model=base_model,
                dataset=dataset,
                created_at=created_at,
                size_mb=round(total_size / (1024 * 1024), 2)
            ))

    return sorted(adapters, key=lambda x: x.created_at or datetime.min, reverse=True)


@router.get("/gguf", response_model=List[GGUFModel])
async def list_gguf_models():
    """List all GGUF models."""
    models = []

    if not os.path.exists(GGUF_DIR):
        os.makedirs(GGUF_DIR, exist_ok=True)
        return models

    for filename in os.listdir(GGUF_DIR):
        if filename.endswith('.gguf'):
            filepath = os.path.join(GGUF_DIR, filename)

            # Extract quantization from filename
            quantization = None
            for q in ['q8_0', 'q6_k', 'q5_k_m', 'q5_k_s', 'q4_k_m', 'q4_k_s', 'q3_k_m', 'q2_k']:
                if q in filename.lower():
                    quantization = q
                    break

            try:
                stat = os.stat(filepath)
                created_at = datetime.fromtimestamp(stat.st_ctime)
                size_mb = round(stat.st_size / (1024 * 1024), 2)
            except Exception:
                created_at = None
                size_mb = 0

            models.append(GGUFModel(
                name=filename.replace('.gguf', ''),
                path=filepath,
                filename=filename,
                size_mb=size_mb,
                quantization=quantization,
                created_at=created_at
            ))

    return sorted(models, key=lambda x: x.created_at or datetime.min, reverse=True)


@router.post("/convert", response_model=ConversionStatusResponse)
async def start_conversion(
    request: ConversionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start a conversion job."""
    service = ModelService(db)

    try:
        job = service.create_conversion_job(
            adapter_path=request.adapter_path,
            adapter_name=request.adapter_name,
            base_model=request.base_model,
            quantization_method=request.quantization_method,
            output_name=request.output_name
        )

        # Start conversion in background
        background_tasks.add_task(service.run_conversion, job.id)

        return job

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/convert/status/{job_id}", response_model=ConversionStatusResponse)
async def get_conversion_status(job_id: int, db: Session = Depends(get_db)):
    """Get conversion job status."""
    job = db.query(ConversionJob).filter(ConversionJob.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Conversion job not found")

    return job


@router.get("/convert/list", response_model=List[ConversionStatusResponse])
async def list_conversion_jobs(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """List conversion jobs."""
    jobs = db.query(ConversionJob).order_by(
        ConversionJob.created_at.desc()
    ).offset(skip).limit(limit).all()

    return jobs


@router.delete("/gguf/{model_name}")
async def delete_gguf_model(model_name: str):
    """Delete a GGUF model."""
    filename = model_name if model_name.endswith('.gguf') else f"{model_name}.gguf"
    filepath = os.path.join(GGUF_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        os.remove(filepath)
        return {"status": "deleted", "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/adapters/{adapter_name}")
async def delete_adapter(adapter_name: str):
    """Delete a LoRA adapter."""
    adapter_dir = os.path.join(OUTPUTS_DIR, adapter_name)

    if not os.path.exists(adapter_dir):
        raise HTTPException(status_code=404, detail="Adapter not found")

    try:
        import shutil
        shutil.rmtree(adapter_dir)
        return {"status": "deleted", "adapter": adapter_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
