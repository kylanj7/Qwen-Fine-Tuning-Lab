"""Evaluation service - wraps evaluate_model.py functionality."""
import os
import sys
import json
import re
import subprocess
from datetime import datetime
from typing import Optional, Dict, List, Any
from sqlalchemy.orm import Session

from ..db.models import EvaluationRun, RunStatus
from ..broadcast_queue import enqueue_broadcast, BroadcastType

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
EVALUATION_RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")


class EvaluationService:
    """Service for managing evaluation runs."""

    # Track running processes
    _processes: Dict[int, subprocess.Popen] = {}
    _log_buffers: Dict[int, List[str]] = {}
    # Track individual question scores for running average calculation
    _question_scores: Dict[int, List[Dict[str, float]]] = {}

    def __init__(self, db: Session):
        self.db = db

    def create_evaluation(
        self,
        model_path: str,
        model_name: str,
        dataset_config_name: str,
        sample_count: int = 10,
        judge_model: str = "nemotron-3-nano:latest"
    ) -> EvaluationRun:
        """Create a new evaluation run."""
        evaluation = EvaluationRun(
            model_path=model_path,
            model_name=model_name,
            dataset_config_name=dataset_config_name,
            sample_count=sample_count,
            judge_model=judge_model,
            status=RunStatus.PENDING,
            total_samples=sample_count
        )

        self.db.add(evaluation)
        self.db.commit()
        self.db.refresh(evaluation)

        return evaluation

    def run_evaluation(self, evaluation_id: int):
        """Run evaluation in a subprocess."""
        from ..db.database import SessionLocal
        db = SessionLocal()

        try:
            evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()
            if not evaluation:
                return

            # Update status
            evaluation.status = RunStatus.RUNNING
            evaluation.started_at = datetime.utcnow()
            db.commit()

            # Build command
            cmd = [
                sys.executable, os.path.join(BASE_DIR, "evaluate_model.py"),
                "--model", evaluation.model_path,
                "--dataset", evaluation.dataset_config_name,
                "--samples", str(evaluation.sample_count),
                "--judge", evaluation.judge_model,
                "--output-json"  # Request JSON output for parsing
            ]

            # Initialize log buffer and score tracking
            EvaluationService._log_buffers[evaluation_id] = []
            EvaluationService._question_scores[evaluation_id] = []

            # Run subprocess with unbuffered output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=BASE_DIR,
                env=env
            )

            EvaluationService._processes[evaluation_id] = process

            # Collect output
            output_lines = []
            print(f"[EvalService] Starting to read subprocess output for eval {evaluation_id}", flush=True)
            for line in process.stdout:
                line = line.strip()
                if line:
                    output_lines.append(line)
                    EvaluationService._log_buffers[evaluation_id].append(line)
                    if len(EvaluationService._log_buffers[evaluation_id]) > 1000:
                        EvaluationService._log_buffers[evaluation_id] = EvaluationService._log_buffers[evaluation_id][-1000:]

                    # Parse progress
                    self._parse_and_update_progress(db, evaluation, line)

            # Wait for completion
            return_code = process.wait()

            # Update final status
            evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()
            if evaluation.status == RunStatus.RUNNING:
                if return_code == 0:
                    evaluation.status = RunStatus.COMPLETED
                    # Parse final results
                    self._parse_final_results(db, evaluation)
                else:
                    evaluation.status = RunStatus.FAILED
                    evaluation.error_message = f"Process exited with code {return_code}"

            evaluation.completed_at = datetime.utcnow()
            db.commit()

            # Broadcast completion
            self._broadcast_complete(evaluation)

        except Exception as e:
            evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()
            if evaluation:
                evaluation.status = RunStatus.FAILED
                evaluation.error_message = str(e)
                evaluation.completed_at = datetime.utcnow()
                db.commit()
        finally:
            EvaluationService._processes.pop(evaluation_id, None)
            EvaluationService._question_scores.pop(evaluation_id, None)
            db.close()

    def cancel_evaluation(self, evaluation_id: int):
        """Cancel an evaluation."""
        evaluation = self.db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()
        if not evaluation:
            return

        if evaluation_id in EvaluationService._processes:
            process = EvaluationService._processes[evaluation_id]
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            EvaluationService._processes.pop(evaluation_id, None)

        evaluation.status = RunStatus.CANCELLED
        evaluation.completed_at = datetime.utcnow()
        self.db.commit()

    def _parse_and_update_progress(self, db: Session, evaluation: EvaluationRun, line: str):
        """Parse evaluation output and update progress."""
        try:
            # Pattern: Evaluating sample X/Y (exact format from evaluate_model.py)
            sample_match = re.search(r"Evaluating sample (\d+)/(\d+)", line)
            if sample_match:
                evaluation.current_sample = int(sample_match.group(1))
                evaluation.total_samples = int(sample_match.group(2))
                db.commit()
                print(f"[EvalService] Parsed progress: {evaluation.current_sample}/{evaluation.total_samples}", flush=True)
                # Broadcast progress update
                self._broadcast_progress(evaluation)
                return  # Don't process further for this line

            # Pattern: SCORES: factual_accuracy=X, completeness=Y, technical_precision=Z
            # (exact format from evaluate_model.py - these are per-question scores)
            score_match = re.search(
                r"SCORES:\s*factual_accuracy=(\d+),\s*completeness=(\d+),\s*technical_precision=(\d+)",
                line
            )
            if score_match:
                factual = float(score_match.group(1))
                complete = float(score_match.group(2))
                precision = float(score_match.group(3))

                # Only include non-zero scores in the running average
                # (zero scores indicate failures/parse errors that should be excluded)
                if factual > 0 or complete > 0 or precision > 0:
                    # Add to tracking list
                    if evaluation.id not in EvaluationService._question_scores:
                        EvaluationService._question_scores[evaluation.id] = []

                    EvaluationService._question_scores[evaluation.id].append({
                        "factual_accuracy": factual,
                        "completeness": complete,
                        "technical_precision": precision,
                    })

                    # Calculate running average from all valid scores
                    scores_list = EvaluationService._question_scores[evaluation.id]
                    n = len(scores_list)
                    evaluation.factual_accuracy = sum(s["factual_accuracy"] for s in scores_list) / n
                    evaluation.completeness = sum(s["completeness"] for s in scores_list) / n
                    evaluation.technical_precision = sum(s["technical_precision"] for s in scores_list) / n
                    # Calculate overall with weights
                    evaluation.overall_score = (
                        evaluation.factual_accuracy * 0.5 +
                        evaluation.completeness * 0.3 +
                        evaluation.technical_precision * 0.2
                    )
                    db.commit()
                    print(f"[EvalService] Running avg ({n} valid): factual={evaluation.factual_accuracy:.1f}, complete={evaluation.completeness:.1f}, precision={evaluation.technical_precision:.1f}", flush=True)
                else:
                    print(f"[EvalService] Skipping zero scores in average calculation", flush=True)

                # Broadcast progress update with scores
                self._broadcast_progress(evaluation)
                return  # Don't process further for this line

            # Try to parse JSON result lines for individual question results
            self._try_parse_question_result(evaluation.id, line)

        except Exception as e:
            # Log parsing errors for debugging
            print(f"[EvalService] Parse error: {e}", flush=True)

    def _broadcast_progress(self, evaluation: EvaluationRun):
        """Broadcast current evaluation progress to WebSocket clients."""
        message = {
            "type": "progress",
            "data": {
                "current_sample": evaluation.current_sample,
                "total_samples": evaluation.total_samples,
                "scores": {
                    "factual_accuracy": evaluation.factual_accuracy,
                    "completeness": evaluation.completeness,
                    "technical_precision": evaluation.technical_precision,
                    "overall_score": evaluation.overall_score,
                }
            }
        }
        print(f"[EvalService] Broadcasting progress for eval {evaluation.id}: sample {evaluation.current_sample}/{evaluation.total_samples}", flush=True)
        enqueue_broadcast(BroadcastType.EVALUATION, evaluation.id, message)

    def _try_parse_question_result(self, evaluation_id: int, line: str):
        """Try to parse a JSON question result line and broadcast it."""
        try:
            # Check if line looks like JSON
            if not line.strip().startswith('{'):
                return

            data = json.loads(line)

            # Check if it has question result structure
            if 'question_idx' in data and 'question' in data:
                print(f"[EvalService] Parsed question result: idx={data.get('question_idx')}", flush=True)
                enqueue_broadcast(
                    BroadcastType.EVALUATION,
                    evaluation_id,
                    {
                        "type": "question_result",
                        "data": {
                            "question_idx": data.get("question_idx"),
                            "question": data.get("question", ""),
                            "model_response": data.get("model_response", ""),
                            "justification": data.get("justification", ""),
                            "scores": data.get("scores", {}),
                            "rag_sources": data.get("rag_sources", []),
                        }
                    }
                )
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"[EvalService] Question result parse error: {e}", flush=True)

    def _broadcast_complete(self, evaluation: EvaluationRun):
        """Broadcast evaluation completion to WebSocket clients."""
        enqueue_broadcast(
            BroadcastType.EVALUATION,
            evaluation.id,
            {
                "type": "complete",
                "data": {
                    "status": evaluation.status.value if hasattr(evaluation.status, 'value') else evaluation.status,
                    "final_scores": {
                        "factual_accuracy": evaluation.factual_accuracy,
                        "completeness": evaluation.completeness,
                        "technical_precision": evaluation.technical_precision,
                        "overall_score": evaluation.overall_score,
                    }
                }
            }
        )

    def _parse_final_results(self, db: Session, evaluation: EvaluationRun):
        """Parse final results from output files."""
        try:
            # Find latest results file
            if not os.path.exists(EVALUATION_RESULTS_DIR):
                return

            # Look for JSON results file
            dataset_name = evaluation.dataset_config_name.lower()
            for filename in sorted(os.listdir(EVALUATION_RESULTS_DIR), reverse=True):
                if filename.startswith(f"eval_{dataset_name}") and filename.endswith(".json"):
                    results_path = os.path.join(EVALUATION_RESULTS_DIR, filename)
                    articles_path = results_path.replace("eval_", "articles_")

                    with open(results_path, 'r') as f:
                        results = json.load(f)

                    # Update scores from "metrics" key (actual JSON structure)
                    if "metrics" in results:
                        metrics = results["metrics"]
                        evaluation.factual_accuracy = metrics.get("factual_accuracy")
                        evaluation.completeness = metrics.get("completeness")
                        evaluation.technical_precision = metrics.get("technical_precision")
                        evaluation.overall_score = metrics.get("overall_accuracy")

                    # Store detailed results from "results" key (actual JSON structure)
                    if "results" in results:
                        detailed = results["results"]

                        # Merge rag_sources from articles file if results lack them
                        if os.path.exists(articles_path):
                            has_rag = any(r.get("rag_sources") for r in detailed)
                            if not has_rag:
                                try:
                                    with open(articles_path, 'r') as af:
                                        articles_data = json.load(af)
                                    article_logs = articles_data.get("article_logs", [])
                                    # Build index-to-sources mapping
                                    sources_by_idx = {}
                                    for entry in article_logs:
                                        q_idx = entry.get("question_index")
                                        papers = entry.get("papers_retrieved", [])
                                        sources_by_idx[q_idx] = [
                                            {
                                                "title": p.get("title", ""),
                                                "year": p.get("year"),
                                                "authors": p.get("authors", []),
                                                "url": p.get("semantic_scholar_url", ""),
                                                "pdf_url": p.get("pdf_url"),
                                                "is_open_access": p.get("is_open_access", False),
                                            }
                                            for p in papers
                                        ]
                                    # Merge into results
                                    for r in detailed:
                                        idx = r.get("index")
                                        if idx in sources_by_idx:
                                            r["rag_sources"] = sources_by_idx[idx]
                                except Exception as e:
                                    print(f"[EvalService] Error merging article logs: {e}")

                        evaluation.detailed_results = detailed

                    evaluation.results_file = results_path
                    if os.path.exists(articles_path):
                        evaluation.articles_file = articles_path

                    db.commit()
                    break

        except Exception as e:
            print(f"Error parsing results: {e}")
