"""Paper PDF download endpoints."""
import os
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..db.models import DownloadedPaper, RunStatus
from ..services.pdf_service import PDFService

router = APIRouter()


class PaperDownloadRequest(BaseModel):
    """Request to download a paper."""
    paper_id: str
    title: str
    pdf_url: str
    authors: List[str] = []
    year: Optional[int] = None
    citation_count: int = 0
    semantic_scholar_url: str = ""
    evaluation_id: Optional[int] = None


class PaperResponse(BaseModel):
    """Paper response."""
    id: int
    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    citation_count: int
    semantic_scholar_url: str
    pdf_url: str
    status: str
    progress: int
    local_path: Optional[str]
    file_size_bytes: Optional[int]
    evaluation_id: Optional[int]
    created_at: datetime
    downloaded_at: Optional[datetime]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class PaperListResponse(BaseModel):
    """List of papers."""
    papers: List[PaperResponse]
    total: int


@router.post("/download", response_model=PaperResponse)
async def queue_paper_download(
    request: PaperDownloadRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Queue a paper for download."""
    service = PDFService(db)

    paper = service.queue_download(
        paper_id=request.paper_id,
        title=request.title,
        pdf_url=request.pdf_url,
        authors=request.authors,
        year=request.year,
        citation_count=request.citation_count,
        semantic_scholar_url=request.semantic_scholar_url,
        evaluation_id=request.evaluation_id,
    )

    # Start download in background if pending
    if paper.status == RunStatus.PENDING:
        background_tasks.add_task(service.download_paper, paper.id)

    return _build_response(paper)


@router.get("/list", response_model=PaperListResponse)
async def list_papers(
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List downloaded papers."""
    service = PDFService(db)
    papers, total = service.list_papers(skip=skip, limit=limit, status=status)

    return PaperListResponse(
        papers=[_build_response(p) for p in papers],
        total=total
    )


@router.get("/{paper_id}", response_model=PaperResponse)
async def get_paper(paper_id: int, db: Session = Depends(get_db)):
    """Get paper details."""
    service = PDFService(db)
    paper = service.get_paper(paper_id)

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    return _build_response(paper)


@router.get("/{paper_id}/file")
async def download_paper_file(paper_id: int, db: Session = Depends(get_db)):
    """Download the actual PDF file."""
    service = PDFService(db)
    paper = service.get_paper(paper_id)

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    if paper.status != RunStatus.COMPLETED or not paper.local_path:
        raise HTTPException(status_code=404, detail="PDF not available")

    if not os.path.exists(paper.local_path):
        raise HTTPException(status_code=404, detail="PDF file not found")

    return FileResponse(
        paper.local_path,
        media_type="application/pdf",
        filename=os.path.basename(paper.local_path)
    )


@router.delete("/{paper_id}")
async def delete_paper(paper_id: int, db: Session = Depends(get_db)):
    """Delete a paper and its file."""
    service = PDFService(db)

    if not service.delete_paper(paper_id):
        raise HTTPException(status_code=404, detail="Paper not found")

    return {"status": "deleted", "paper_id": paper_id}


@router.post("/{paper_id}/retry", response_model=PaperResponse)
async def retry_download(
    paper_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Retry a failed download."""
    service = PDFService(db)
    paper = service.retry_failed(paper_id)

    if not paper:
        raise HTTPException(status_code=400, detail="Paper not found or not failed")

    # Start download in background
    background_tasks.add_task(service.download_paper, paper.id)

    return _build_response(paper)


@router.delete("/clear/all")
async def clear_all_papers(db: Session = Depends(get_db)):
    """Clear all papers (except currently downloading)."""
    service = PDFService(db)
    count = service.clear_all()

    return {"status": "cleared", "deleted_count": count}


def _build_response(paper: DownloadedPaper) -> PaperResponse:
    """Build paper response."""
    return PaperResponse(
        id=paper.id,
        paper_id=paper.paper_id,
        title=paper.title,
        authors=paper.authors or [],
        year=paper.year,
        citation_count=paper.citation_count,
        semantic_scholar_url=paper.semantic_scholar_url,
        pdf_url=paper.pdf_url,
        status=paper.status,
        progress=paper.progress,
        local_path=paper.local_path,
        file_size_bytes=paper.file_size_bytes,
        evaluation_id=paper.evaluation_id,
        created_at=paper.created_at,
        downloaded_at=paper.downloaded_at,
        error_message=paper.error_message,
    )
