"""Evaluation endpoints."""
import os
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..db.models import EvaluationRun, RunStatus
from ..services.evaluation_service import EvaluationService

router = APIRouter()


class EvaluationStartRequest(BaseModel):
    """Request to start an evaluation."""
    model_path: str
    model_name: str
    dataset_config_name: str
    sample_count: int = 10
    judge_model: str = "nemotron-3-nano:latest"


class EvaluationScores(BaseModel):
    """Evaluation scores response."""
    factual_accuracy: Optional[float]
    completeness: Optional[float]
    technical_precision: Optional[float]
    overall_score: Optional[float]


class EvaluationStatusResponse(BaseModel):
    """Evaluation run status response."""
    id: int
    model_path: str
    model_name: str
    dataset_config_name: str
    sample_count: int
    judge_model: str
    status: str
    current_sample: int
    total_samples: int
    scores: Optional[EvaluationScores]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class EvaluationDetailedResult(BaseModel):
    """Detailed per-question result."""
    question_idx: int
    question: str
    model_response: str
    justification: Optional[str] = None
    scores: EvaluationScores
    rag_sources: List[dict] = []


class EvaluationResultsResponse(BaseModel):
    """Full evaluation results."""
    id: int
    model_name: str
    status: str
    overall_scores: Optional[EvaluationScores]
    detailed_results: List[EvaluationDetailedResult]
    results_file: Optional[str]
    articles_file: Optional[str]


class EvaluationListResponse(BaseModel):
    """List of evaluation runs."""
    evaluations: List[EvaluationStatusResponse]
    total: int


@router.post("/start", response_model=EvaluationStatusResponse)
async def start_evaluation(
    request: EvaluationStartRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start a new evaluation."""
    service = EvaluationService(db)

    try:
        evaluation = service.create_evaluation(
            model_path=request.model_path,
            model_name=request.model_name,
            dataset_config_name=request.dataset_config_name,
            sample_count=request.sample_count,
            judge_model=request.judge_model
        )

        # Start evaluation in background
        background_tasks.add_task(service.run_evaluation, evaluation.id)

        # Build response with scores
        return _build_status_response(evaluation)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{evaluation_id}", response_model=EvaluationStatusResponse)
async def get_evaluation_status(evaluation_id: int, db: Session = Depends(get_db)):
    """Get evaluation status."""
    evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()

    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return _build_status_response(evaluation)


@router.get("/results/{evaluation_id}", response_model=EvaluationResultsResponse)
async def get_evaluation_results(evaluation_id: int, db: Session = Depends(get_db)):
    """Get full evaluation results."""
    evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()

    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    overall_scores = None
    if evaluation.overall_score is not None:
        overall_scores = EvaluationScores(
            factual_accuracy=evaluation.factual_accuracy,
            completeness=evaluation.completeness,
            technical_precision=evaluation.technical_precision,
            overall_score=evaluation.overall_score
        )

    detailed_results = []
    if evaluation.detailed_results:
        # If results lack rag_sources, try to load from articles file
        sources_by_idx = {}
        has_rag = any(r.get("rag_sources") for r in evaluation.detailed_results)
        if not has_rag and evaluation.articles_file and os.path.exists(evaluation.articles_file):
            try:
                import json
                with open(evaluation.articles_file, 'r') as af:
                    articles_data = json.load(af)
                for entry in articles_data.get("article_logs", []):
                    q_idx = entry.get("question_index")
                    sources_by_idx[q_idx] = [
                        {
                            "title": p.get("title", ""),
                            "year": p.get("year"),
                            "authors": p.get("authors", []),
                            "url": p.get("semantic_scholar_url", ""),
                            "pdf_url": p.get("pdf_url"),
                            "is_open_access": p.get("is_open_access", False),
                        }
                        for p in entry.get("papers_retrieved", [])
                    ]
            except Exception:
                pass

        for result in evaluation.detailed_results:
            # Handle both flat and nested score formats
            scores_data = result.get("scores", {})
            if not scores_data:
                # Fallback to flat format
                scores_data = {
                    "factual_accuracy": result.get("factual_accuracy"),
                    "completeness": result.get("completeness"),
                    "technical_precision": result.get("technical_precision"),
                    "overall_score": result.get("overall_score"),
                }

            # Handle both 0-indexed question_idx and 1-indexed index
            idx = result.get("question_idx")
            if idx is None:
                idx = result.get("index", 1) - 1  # Convert 1-based index to 0-based

            # Get rag_sources from result, or fall back to articles file
            rag_sources = result.get("rag_sources", [])
            if not rag_sources:
                # index in the result is 1-based, matching question_index in articles
                result_index = result.get("index", idx + 1)
                rag_sources = sources_by_idx.get(result_index, [])

            detailed_results.append(EvaluationDetailedResult(
                question_idx=idx,
                question=result.get("question", ""),
                model_response=result.get("model_response", result.get("answer", "")),
                justification=result.get("justification", ""),
                scores=EvaluationScores(
                    factual_accuracy=scores_data.get("factual_accuracy"),
                    completeness=scores_data.get("completeness"),
                    technical_precision=scores_data.get("technical_precision"),
                    overall_score=scores_data.get("overall_score", result.get("overall_score"))
                ),
                rag_sources=rag_sources,
            ))

    return EvaluationResultsResponse(
        id=evaluation.id,
        model_name=evaluation.model_name,
        status=evaluation.status,
        overall_scores=overall_scores,
        detailed_results=detailed_results,
        results_file=evaluation.results_file,
        articles_file=evaluation.articles_file
    )


@router.get("/list", response_model=EvaluationListResponse)
async def list_evaluations(
    skip: int = 0,
    limit: int = 20,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List evaluations with pagination."""
    query = db.query(EvaluationRun)

    if status:
        query = query.filter(EvaluationRun.status == status)

    total = query.count()
    evaluations = query.order_by(EvaluationRun.created_at.desc()).offset(skip).limit(limit).all()

    return EvaluationListResponse(
        evaluations=[_build_status_response(e) for e in evaluations],
        total=total
    )


@router.post("/cancel/{evaluation_id}")
async def cancel_evaluation(evaluation_id: int, db: Session = Depends(get_db)):
    """Cancel an evaluation."""
    evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()

    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    if evaluation.status not in [RunStatus.PENDING, RunStatus.RUNNING]:
        raise HTTPException(status_code=400, detail="Can only cancel pending or running evaluations")

    service = EvaluationService(db)
    service.cancel_evaluation(evaluation_id)

    return {"status": "cancelled", "evaluation_id": evaluation_id}


@router.delete("/clear")
async def clear_evaluations(db: Session = Depends(get_db)):
    """Clear all completed and failed evaluations."""
    # Only delete evaluations that are not currently running
    deleted_count = db.query(EvaluationRun).filter(
        EvaluationRun.status.notin_([RunStatus.PENDING, RunStatus.RUNNING])
    ).delete(synchronize_session=False)
    db.commit()

    return {"status": "cleared", "deleted_count": deleted_count}


def _build_status_response(evaluation: EvaluationRun) -> EvaluationStatusResponse:
    """Build status response from evaluation model."""
    scores = None
    if evaluation.overall_score is not None:
        scores = EvaluationScores(
            factual_accuracy=evaluation.factual_accuracy,
            completeness=evaluation.completeness,
            technical_precision=evaluation.technical_precision,
            overall_score=evaluation.overall_score
        )

    return EvaluationStatusResponse(
        id=evaluation.id,
        model_path=evaluation.model_path,
        model_name=evaluation.model_name,
        dataset_config_name=evaluation.dataset_config_name,
        sample_count=evaluation.sample_count,
        judge_model=evaluation.judge_model,
        status=evaluation.status,
        current_sample=evaluation.current_sample,
        total_samples=evaluation.total_samples,
        scores=scores,
        created_at=evaluation.created_at,
        started_at=evaluation.started_at,
        completed_at=evaluation.completed_at,
        error_message=evaluation.error_message
    )
