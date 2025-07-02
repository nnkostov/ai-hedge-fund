from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from app.backend.models.schemas import ErrorResponse
from src.llm.models import get_models_list

router = APIRouter(prefix="/language-models")

@router.get(
    path="/",
    responses={
        200: {"description": "List of available language models"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_language_models():
    """Get the list of available cloud-based language models."""
    try:
        return {"models": get_models_list()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {str(e)}")

@router.get(
    path="/providers",
    responses={
        200: {"description": "List of available model providers"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_language_model_providers():
    """Get the list of available model providers with their models grouped."""
    try:
        models = get_models_list()
        
        # Group models by provider
        providers = {}
        for model in models:
            provider_name = model["provider"]
            if provider_name not in providers:
                providers[provider_name] = {
                    "name": provider_name,
                    "models": []
                }
            providers[provider_name]["models"].append({
                "display_name": model["display_name"],
                "model_name": model["model_name"]
            })
        
        return {"providers": list(providers.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve providers: {str(e)}") 