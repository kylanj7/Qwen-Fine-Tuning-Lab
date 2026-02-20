"""SQLAlchemy models for tracking runs and evaluations."""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from .database import Base


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingRun(Base):
    """Training run record."""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_name = Column(String(255), unique=True, index=True)

    # Configuration
    model_config_name = Column(String(255))
    dataset_config_name = Column(String(255))
    training_config_name = Column(String(255))

    # Override parameters (JSON)
    parameter_overrides = Column(JSON, default={})

    # Status tracking
    status = Column(String(50), default=RunStatus.PENDING)
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    current_epoch = Column(Float, default=0.0)
    total_epochs = Column(Integer, default=0)

    # Metrics
    current_loss = Column(Float, nullable=True)
    best_loss = Column(Float, nullable=True)

    # WandB integration
    wandb_project = Column(String(255), nullable=True)
    wandb_run_url = Column(String(512), nullable=True)

    # Paths
    output_dir = Column(String(512), nullable=True)
    adapter_path = Column(String(512), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)

    # Relationships
    evaluations = relationship("EvaluationRun", back_populates="training_run")


class EvaluationRun(Base):
    """Evaluation run record."""
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)

    # Link to training run (optional - can evaluate external models)
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=True)

    # Model info
    model_path = Column(String(512))
    model_name = Column(String(255))

    # Dataset info
    dataset_config_name = Column(String(255))
    sample_count = Column(Integer, default=10)

    # Judge model
    judge_model = Column(String(255), default="nemotron-3-nano:latest")

    # Status tracking
    status = Column(String(50), default=RunStatus.PENDING)
    current_sample = Column(Integer, default=0)
    total_samples = Column(Integer, default=0)

    # Scores (3-dimension)
    factual_accuracy = Column(Float, nullable=True)
    completeness = Column(Float, nullable=True)
    technical_precision = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)

    # Detailed results (JSON array of per-question scores)
    detailed_results = Column(JSON, default=[])

    # Paths
    results_file = Column(String(512), nullable=True)
    articles_file = Column(String(512), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)

    # Relationships
    training_run = relationship("TrainingRun", back_populates="evaluations")


class ConversionJob(Base):
    """Model conversion job record."""
    __tablename__ = "conversion_jobs"

    id = Column(Integer, primary_key=True, index=True)

    # Source info
    adapter_path = Column(String(512))
    adapter_name = Column(String(255))
    base_model = Column(String(255))

    # Conversion settings
    quantization_method = Column(String(50), default="q4_k_m")
    output_name = Column(String(255), nullable=True)

    # Status tracking
    status = Column(String(50), default=RunStatus.PENDING)
    current_stage = Column(String(100), default="")  # merge, convert, quantize
    progress = Column(Integer, default=0)  # 0-100

    # Output
    output_path = Column(String(512), nullable=True)
    output_size_bytes = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)


class DownloadedPaper(Base):
    """Downloaded paper PDF record."""
    __tablename__ = "downloaded_papers"

    id = Column(Integer, primary_key=True, index=True)

    # Paper info from Semantic Scholar
    paper_id = Column(String(255), unique=True, index=True)
    title = Column(String(1024))
    authors = Column(JSON, default=[])  # List of author names
    year = Column(Integer, nullable=True)
    citation_count = Column(Integer, default=0)

    # URLs
    semantic_scholar_url = Column(String(512))
    pdf_url = Column(String(512))  # Original PDF URL

    # Download status
    status = Column(String(50), default=RunStatus.PENDING)
    progress = Column(Integer, default=0)  # 0-100

    # Local storage
    local_path = Column(String(512), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)

    # Source evaluation (which evaluation triggered this download)
    evaluation_id = Column(Integer, ForeignKey("evaluation_runs.id"), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    downloaded_at = Column(DateTime, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)


class ChatSession(Base):
    """Chat session record."""
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)

    # Model info
    model_path = Column(String(512))
    model_name = Column(String(255))

    # Settings
    system_prompt = Column(Text, default="")
    temperature = Column(Float, default=0.7)
    top_p = Column(Float, default=0.9)
    top_k = Column(Integer, default=40)
    max_tokens = Column(Integer, default=2048)

    # Messages (JSON array)
    messages = Column(JSON, default=[])

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
