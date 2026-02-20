"""PDF download service - downloads papers from Semantic Scholar."""
import os
import requests
from datetime import datetime
from typing import Optional, List, Dict
from sqlalchemy.orm import Session

from ..db.models import DownloadedPaper, RunStatus

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
PDF_DIR = os.path.join(BASE_DIR, "papers")


class PDFService:
    """Service for downloading and managing paper PDFs."""

    def __init__(self, db: Session):
        self.db = db
        # Ensure PDF directory exists
        os.makedirs(PDF_DIR, exist_ok=True)

    def queue_download(
        self,
        paper_id: str,
        title: str,
        pdf_url: str,
        authors: List[str] = None,
        year: Optional[int] = None,
        citation_count: int = 0,
        semantic_scholar_url: str = "",
        evaluation_id: Optional[int] = None,
    ) -> DownloadedPaper:
        """Queue a paper for download. Returns existing record if already queued/downloaded."""
        # Check if already exists
        existing = self.db.query(DownloadedPaper).filter(
            DownloadedPaper.paper_id == paper_id
        ).first()

        if existing:
            return existing

        paper = DownloadedPaper(
            paper_id=paper_id,
            title=title,
            authors=authors or [],
            year=year,
            citation_count=citation_count,
            semantic_scholar_url=semantic_scholar_url,
            pdf_url=pdf_url,
            status=RunStatus.PENDING,
            evaluation_id=evaluation_id,
        )

        self.db.add(paper)
        self.db.commit()
        self.db.refresh(paper)

        return paper

    def download_paper(self, paper_id: int):
        """Download a paper PDF. Call from background task."""
        from ..db.database import SessionLocal
        db = SessionLocal()

        try:
            paper = db.query(DownloadedPaper).filter(DownloadedPaper.id == paper_id).first()
            if not paper:
                return

            if paper.status == RunStatus.COMPLETED:
                return  # Already downloaded

            # Update status
            paper.status = RunStatus.RUNNING
            paper.progress = 10
            db.commit()

            # Create safe filename
            safe_title = "".join(c for c in paper.title[:80] if c.isalnum() or c in " -_").strip()
            filename = f"{paper.paper_id}_{safe_title}.pdf"
            local_path = os.path.join(PDF_DIR, filename)

            # Download with streaming
            try:
                response = requests.get(
                    paper.pdf_url,
                    stream=True,
                    timeout=60,
                    headers={"User-Agent": "QwenFineTuneTestSuite/1.0"}
                )
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                paper.progress = min(90, int(10 + (downloaded / total_size) * 80))
                                db.commit()

                # Success
                paper.status = RunStatus.COMPLETED
                paper.progress = 100
                paper.local_path = local_path
                paper.file_size_bytes = os.path.getsize(local_path)
                paper.downloaded_at = datetime.utcnow()
                db.commit()

            except requests.RequestException as e:
                paper.status = RunStatus.FAILED
                paper.error_message = str(e)
                db.commit()

        except Exception as e:
            paper = db.query(DownloadedPaper).filter(DownloadedPaper.id == paper_id).first()
            if paper:
                paper.status = RunStatus.FAILED
                paper.error_message = str(e)
                db.commit()
        finally:
            db.close()

    def list_papers(
        self,
        skip: int = 0,
        limit: int = 50,
        status: Optional[str] = None
    ) -> tuple[List[DownloadedPaper], int]:
        """List downloaded papers with pagination."""
        query = self.db.query(DownloadedPaper)

        if status:
            query = query.filter(DownloadedPaper.status == status)

        total = query.count()
        papers = query.order_by(DownloadedPaper.created_at.desc()).offset(skip).limit(limit).all()

        return papers, total

    def get_paper(self, paper_id: int) -> Optional[DownloadedPaper]:
        """Get a paper by ID."""
        return self.db.query(DownloadedPaper).filter(DownloadedPaper.id == paper_id).first()

    def delete_paper(self, paper_id: int) -> bool:
        """Delete a paper and its file."""
        paper = self.db.query(DownloadedPaper).filter(DownloadedPaper.id == paper_id).first()

        if not paper:
            return False

        # Delete file if exists
        if paper.local_path and os.path.exists(paper.local_path):
            try:
                os.remove(paper.local_path)
            except OSError:
                pass

        self.db.delete(paper)
        self.db.commit()
        return True

    def clear_all(self) -> int:
        """Clear all papers (except currently downloading)."""
        # Get papers to delete
        papers = self.db.query(DownloadedPaper).filter(
            DownloadedPaper.status != RunStatus.RUNNING
        ).all()

        count = 0
        for paper in papers:
            if paper.local_path and os.path.exists(paper.local_path):
                try:
                    os.remove(paper.local_path)
                except OSError:
                    pass
            self.db.delete(paper)
            count += 1

        self.db.commit()
        return count

    def get_pending_downloads(self) -> List[DownloadedPaper]:
        """Get papers waiting to be downloaded."""
        return self.db.query(DownloadedPaper).filter(
            DownloadedPaper.status == RunStatus.PENDING
        ).all()

    def retry_failed(self, paper_id: int) -> Optional[DownloadedPaper]:
        """Retry a failed download."""
        paper = self.db.query(DownloadedPaper).filter(DownloadedPaper.id == paper_id).first()

        if not paper or paper.status != RunStatus.FAILED:
            return None

        paper.status = RunStatus.PENDING
        paper.progress = 0
        paper.error_message = None
        self.db.commit()
        self.db.refresh(paper)

        return paper
