"""Training service - wraps train.py functionality."""
import os
import sys
import json
import yaml
import asyncio
import subprocess
import threading
from datetime import datetime
from typing import Optional, Dict, List
from sqlalchemy.orm import Session

from ..db.models import TrainingRun, RunStatus

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


class TrainingService:
    """Service for managing training runs."""

    # Track running processes by run_id
    _processes: Dict[int, subprocess.Popen] = {}
    _log_buffers: Dict[int, List[str]] = {}

    def __init__(self, db: Session):
        self.db = db

    def create_run(
        self,
        model_config_name: str,
        dataset_config_name: str,
        training_config_name: str,
        parameter_overrides: Optional[dict] = None
    ) -> TrainingRun:
        """Create a new training run record."""
        # Load configs to get additional info
        model_config = self._load_config("models", model_config_name)
        dataset_config = self._load_config("datasets", dataset_config_name)
        training_config = self._load_config("training", training_config_name)

        # Generate run name
        run_name = self._generate_run_name(model_config, dataset_config)

        # Calculate total epochs
        total_epochs = training_config.get("num_train_epochs", 3)
        if parameter_overrides and "num_train_epochs" in parameter_overrides:
            total_epochs = parameter_overrides["num_train_epochs"]

        # Create WandB project name
        wandb_template = training_config.get("wandb_project_template", "{dataset_name}-{model_name}")
        wandb_project = wandb_template.format(
            dataset_name=dataset_config.get("name", "dataset"),
            model_name=model_config.get("name", "model")
        )

        run = TrainingRun(
            run_name=run_name,
            model_config_name=model_config_name,
            dataset_config_name=dataset_config_name,
            training_config_name=training_config_name,
            parameter_overrides=parameter_overrides or {},
            status=RunStatus.PENDING,
            total_epochs=total_epochs,
            wandb_project=wandb_project,
            output_dir=os.path.join(OUTPUTS_DIR, run_name)
        )

        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)

        return run

    def run_training(self, run_id: int):
        """Run training in a subprocess."""
        # Get fresh session for background thread
        from ..db.database import SessionLocal
        db = SessionLocal()

        try:
            run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if not run:
                return

            # Update status
            run.status = RunStatus.RUNNING
            run.started_at = datetime.utcnow()
            db.commit()

            # Build command
            cmd = [
                sys.executable, os.path.join(BASE_DIR, "train.py"),
                "--model", run.model_config_name,
                "--dataset", run.dataset_config_name,
                "--training", run.training_config_name,
                "--run-name", run.run_name
            ]

            # Add parameter overrides
            if run.parameter_overrides:
                for key, value in run.parameter_overrides.items():
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])

            # Initialize log buffer
            TrainingService._log_buffers[run_id] = []

            # Run subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=BASE_DIR
            )

            TrainingService._processes[run_id] = process

            # Stream output and parse progress
            for line in process.stdout:
                line = line.strip()
                if line:
                    # Add to log buffer (keep last 1000 lines)
                    TrainingService._log_buffers[run_id].append(line)
                    if len(TrainingService._log_buffers[run_id]) > 1000:
                        TrainingService._log_buffers[run_id] = TrainingService._log_buffers[run_id][-1000:]

                    # Parse progress from output
                    self._parse_and_update_progress(db, run, line)

            # Wait for completion
            return_code = process.wait()

            # Update final status
            run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run.status == RunStatus.RUNNING:  # Not cancelled
                if return_code == 0:
                    run.status = RunStatus.COMPLETED
                    run.adapter_path = os.path.join(run.output_dir, "final_adapter")
                else:
                    run.status = RunStatus.FAILED
                    run.error_message = f"Process exited with code {return_code}"

            run.completed_at = datetime.utcnow()
            db.commit()

        except Exception as e:
            run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run:
                run.status = RunStatus.FAILED
                run.error_message = str(e)
                run.completed_at = datetime.utcnow()
                db.commit()
        finally:
            # Cleanup
            TrainingService._processes.pop(run_id, None)
            db.close()

    def cancel_run(self, run_id: int):
        """Cancel a running training."""
        run = self.db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if not run:
            return

        # Kill process if running
        if run_id in TrainingService._processes:
            process = TrainingService._processes[run_id]
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            TrainingService._processes.pop(run_id, None)

        # Update status
        run.status = RunStatus.CANCELLED
        run.completed_at = datetime.utcnow()
        self.db.commit()

    def get_logs(self, run_id: int, lines: int = 100) -> List[str]:
        """Get recent training logs."""
        if run_id in TrainingService._log_buffers:
            return TrainingService._log_buffers[run_id][-lines:]

        # Try to read from log file if process not running
        run = self.db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if run and run.output_dir:
            log_file = os.path.join(run.output_dir, "training.log")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    return f.readlines()[-lines:]

        return []

    def _load_config(self, config_type: str, config_name: str) -> dict:
        """Load a configuration file."""
        # Find config file
        config_dir = os.path.join(CONFIGS_DIR, config_type)
        config_file = None

        for filename in os.listdir(config_dir):
            if filename.endswith(('.yaml', '.yml')):
                filepath = os.path.join(config_dir, filename)
                with open(filepath, 'r') as f:
                    config = yaml.safe_load(f)
                if config.get('name') == config_name or filename.replace('.yaml', '').replace('.yml', '') == config_name:
                    return config

        raise ValueError(f"Config not found: {config_type}/{config_name}")

    def _generate_run_name(self, model_config: dict, dataset_config: dict) -> str:
        """Generate a unique run name."""
        # Load or create training index
        index_file = os.path.join(BASE_DIR, ".training_index.json")
        index = {}
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index = json.load(f)

        # Get current date
        date_str = datetime.now().strftime("%Y%m%d")

        # Build base name
        model_name = model_config.get("name", "model").lower().replace(" ", "-")
        model_size = model_config.get("size", "").lower()
        dataset_name = dataset_config.get("name", "dataset").lower().replace(" ", "-")

        base_name = f"{model_name}-{model_size}-{dataset_name}-{date_str}"

        # Get next index
        run_index = index.get(base_name, 0) + 1
        index[base_name] = run_index

        # Save index
        with open(index_file, 'w') as f:
            json.dump(index, f)

        return f"{base_name}-{run_index:03d}"

    def _parse_and_update_progress(self, db: Session, run: TrainingRun, line: str):
        """Parse training output and update progress."""
        try:
            # Look for progress patterns
            # Pattern: {'loss': 0.5, 'grad_norm': 1.0, 'learning_rate': 1e-5, 'epoch': 0.5}
            if "'loss':" in line and "'epoch':" in line:
                # Parse training metrics
                import re
                loss_match = re.search(r"'loss':\s*([\d.]+)", line)
                epoch_match = re.search(r"'epoch':\s*([\d.]+)", line)

                if loss_match:
                    run.current_loss = float(loss_match.group(1))
                    if run.best_loss is None or run.current_loss < run.best_loss:
                        run.best_loss = run.current_loss

                if epoch_match:
                    run.current_epoch = float(epoch_match.group(1))

            # Pattern: Step X/Y
            if "Step" in line and "/" in line:
                import re
                step_match = re.search(r"Step\s+(\d+)/(\d+)", line)
                if step_match:
                    run.current_step = int(step_match.group(1))
                    run.total_steps = int(step_match.group(2))

            # Pattern for tqdm progress: X%|...| Y/Z
            if "|" in line and "%" in line:
                import re
                progress_match = re.search(r"(\d+)%\|.*\|\s*(\d+)/(\d+)", line)
                if progress_match:
                    run.current_step = int(progress_match.group(2))
                    run.total_steps = int(progress_match.group(3))

            # WandB URL pattern
            if "wandb:" in line.lower() and "http" in line:
                import re
                url_match = re.search(r"(https?://wandb\.ai/\S+)", line)
                if url_match:
                    run.wandb_run_url = url_match.group(1)

            db.commit()

        except Exception:
            # Don't crash on parse errors
            pass
