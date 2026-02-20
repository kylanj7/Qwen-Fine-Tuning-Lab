"""Configuration discovery endpoints."""
import os
import yaml
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")


class ConfigItem(BaseModel):
    """Configuration item response."""
    name: str
    filename: str
    path: str
    description: Optional[str] = None


class ConfigDetail(BaseModel):
    """Full configuration details."""
    name: str
    filename: str
    path: str
    content: dict


class ConfigListResponse(BaseModel):
    """List of configurations."""
    type: str
    configs: List[ConfigItem]


def discover_configs(config_type: str) -> List[ConfigItem]:
    """Discover configuration files of a given type."""
    config_path = os.path.join(CONFIGS_DIR, config_type)

    if not os.path.exists(config_path):
        return []

    configs = []
    for filename in sorted(os.listdir(config_path)):
        if filename.endswith(('.yaml', '.yml')):
            filepath = os.path.join(config_path, filename)
            try:
                with open(filepath, 'r') as f:
                    content = yaml.safe_load(f)

                name = content.get('name', filename.replace('.yaml', '').replace('.yml', ''))
                description = content.get('description', None)

                configs.append(ConfigItem(
                    name=name,
                    filename=filename,
                    path=filepath,
                    description=description
                ))
            except Exception as e:
                print(f"Error loading config {filepath}: {e}")

    return configs


@router.get("/models", response_model=ConfigListResponse)
async def list_model_configs():
    """List available model configurations."""
    configs = discover_configs("models")
    return ConfigListResponse(type="models", configs=configs)


@router.get("/datasets", response_model=ConfigListResponse)
async def list_dataset_configs():
    """List available dataset configurations."""
    configs = discover_configs("datasets")
    return ConfigListResponse(type="datasets", configs=configs)


@router.get("/training", response_model=ConfigListResponse)
async def list_training_configs():
    """List available training configurations."""
    configs = discover_configs("training")
    return ConfigListResponse(type="training", configs=configs)


@router.get("/{config_type}/{filename}", response_model=ConfigDetail)
async def get_config_detail(config_type: str, filename: str):
    """Get full configuration details."""
    if config_type not in ["models", "datasets", "training"]:
        raise HTTPException(status_code=400, detail="Invalid config type")

    filepath = os.path.join(CONFIGS_DIR, config_type, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Config not found")

    try:
        with open(filepath, 'r') as f:
            content = yaml.safe_load(f)

        name = content.get('name', filename.replace('.yaml', '').replace('.yml', ''))

        return ConfigDetail(
            name=name,
            filename=filename,
            path=filepath,
            content=content
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading config: {e}")


@router.get("/all")
async def list_all_configs():
    """List all available configurations."""
    return {
        "models": discover_configs("models"),
        "datasets": discover_configs("datasets"),
        "training": discover_configs("training")
    }
