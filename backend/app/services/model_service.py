"""Model service - wraps merge_and_convert_gguff.py functionality."""
import os
import sys
import json
import subprocess
from datetime import datetime
from typing import Optional, Dict, List
from sqlalchemy.orm import Session

from ..db.models import ConversionJob, RunStatus

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
GGUF_DIR = os.path.join(BASE_DIR, "models", "gguf")


class ModelService:
    """Service for managing model conversions."""

    # Track running processes
    _processes: Dict[int, subprocess.Popen] = {}
    _log_buffers: Dict[int, List[str]] = {}

    def __init__(self, db: Session):
        self.db = db

    def create_conversion_job(
        self,
        adapter_path: str,
        adapter_name: str,
        base_model: Optional[str] = None,
        quantization_method: str = "q4_k_m",
        output_name: Optional[str] = None
    ) -> ConversionJob:
        """Create a new conversion job."""
        # Try to auto-detect base model from metadata
        if not base_model:
            metadata_path = os.path.join(os.path.dirname(adapter_path), "run_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                base_model = metadata.get("base_model", "Qwen/Qwen2.5-14B-Instruct")
            else:
                base_model = "Qwen/Qwen2.5-14B-Instruct"

        job = ConversionJob(
            adapter_path=adapter_path,
            adapter_name=adapter_name,
            base_model=base_model,
            quantization_method=quantization_method,
            output_name=output_name,
            status=RunStatus.PENDING
        )

        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        return job

    def run_conversion(self, job_id: int):
        """Run conversion in a subprocess."""
        from ..db.database import SessionLocal
        db = SessionLocal()

        try:
            job = db.query(ConversionJob).filter(ConversionJob.id == job_id).first()
            if not job:
                return

            # Update status
            job.status = RunStatus.RUNNING
            job.started_at = datetime.utcnow()
            job.current_stage = "Starting"
            db.commit()

            # Build command
            cmd = [
                sys.executable, os.path.join(BASE_DIR, "merge_and_convert_gguff.py"),
                "--adapter-path", job.adapter_path,
                "--base-model", job.base_model,
                "--quantization", job.quantization_method
            ]

            if job.output_name:
                cmd.extend(["--output-name", job.output_name])

            # Initialize log buffer
            ModelService._log_buffers[job_id] = []

            # Run subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=BASE_DIR
            )

            ModelService._processes[job_id] = process

            # Stream output
            for line in process.stdout:
                line = line.strip()
                if line:
                    ModelService._log_buffers[job_id].append(line)
                    if len(ModelService._log_buffers[job_id]) > 1000:
                        ModelService._log_buffers[job_id] = ModelService._log_buffers[job_id][-1000:]

                    # Parse progress
                    self._parse_and_update_progress(db, job, line)

            # Wait for completion
            return_code = process.wait()

            # Update final status
            job = db.query(ConversionJob).filter(ConversionJob.id == job_id).first()
            if job.status == RunStatus.RUNNING:
                if return_code == 0:
                    job.status = RunStatus.COMPLETED
                    job.current_stage = "Complete"
                    job.progress = 100
                    # Find output file
                    self._find_output_file(job)
                else:
                    job.status = RunStatus.FAILED
                    job.error_message = f"Process exited with code {return_code}"

            job.completed_at = datetime.utcnow()
            db.commit()

        except Exception as e:
            job = db.query(ConversionJob).filter(ConversionJob.id == job_id).first()
            if job:
                job.status = RunStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                db.commit()
        finally:
            ModelService._processes.pop(job_id, None)
            db.close()

    def cancel_conversion(self, job_id: int):
        """Cancel a conversion job."""
        job = self.db.query(ConversionJob).filter(ConversionJob.id == job_id).first()
        if not job:
            return

        if job_id in ModelService._processes:
            process = ModelService._processes[job_id]
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            ModelService._processes.pop(job_id, None)

        job.status = RunStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        self.db.commit()

    def get_logs(self, job_id: int, lines: int = 100) -> List[str]:
        """Get recent conversion logs."""
        if job_id in ModelService._log_buffers:
            return ModelService._log_buffers[job_id][-lines:]
        return []

    def _parse_and_update_progress(self, db: Session, job: ConversionJob, line: str):
        """Parse conversion output and update progress."""
        try:
            line_lower = line.lower()

            # Stage detection
            if "merging" in line_lower or "merge" in line_lower:
                job.current_stage = "Merging LoRA"
                job.progress = 20
            elif "converting" in line_lower and "gguf" in line_lower:
                job.current_stage = "Converting to GGUF"
                job.progress = 50
            elif "quantiz" in line_lower:
                job.current_stage = "Quantizing"
                job.progress = 75
            elif "cleaning" in line_lower:
                job.current_stage = "Cleaning up"
                job.progress = 90
            elif "complete" in line_lower or "success" in line_lower or "saved" in line_lower:
                job.current_stage = "Complete"
                job.progress = 100

            # Progress percentage detection
            import re
            progress_match = re.search(r"(\d+)%", line)
            if progress_match:
                detected_progress = int(progress_match.group(1))
                # Scale based on current stage
                if job.current_stage == "Merging LoRA":
                    job.progress = 20 + (detected_progress * 0.3)
                elif job.current_stage == "Converting to GGUF":
                    job.progress = 50 + (detected_progress * 0.25)
                elif job.current_stage == "Quantizing":
                    job.progress = 75 + (detected_progress * 0.15)

            db.commit()

        except Exception:
            pass

    def _find_output_file(self, job: ConversionJob):
        """Find the output GGUF file."""
        if not os.path.exists(GGUF_DIR):
            return

        # Expected filename pattern
        adapter_name = job.adapter_name.replace("_", "-")
        quant = job.quantization_method.upper().replace("_", "-")

        for filename in os.listdir(GGUF_DIR):
            if adapter_name in filename or (job.output_name and job.output_name in filename):
                if filename.endswith('.gguf'):
                    filepath = os.path.join(GGUF_DIR, filename)
                    job.output_path = filepath
                    job.output_size_bytes = os.path.getsize(filepath)
                    return

        # Fallback: find most recent GGUF file
        gguf_files = [
            (f, os.path.getctime(os.path.join(GGUF_DIR, f)))
            for f in os.listdir(GGUF_DIR)
            if f.endswith('.gguf')
        ]
        if gguf_files:
            latest = max(gguf_files, key=lambda x: x[1])
            filepath = os.path.join(GGUF_DIR, latest[0])
            job.output_path = filepath
            job.output_size_bytes = os.path.getsize(filepath)
