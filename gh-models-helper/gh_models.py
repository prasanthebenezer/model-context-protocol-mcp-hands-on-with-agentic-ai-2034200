# model_comparison_server.py
import os
from typing import List, Dict, Any, Optional
import asyncio
import json
import requests

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from mcp.server.fastmcp import FastMCP, Context

from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

# Create an MCP server
mcp = FastMCP("GitHub Models Comparison")

# Azure AI configuration
ENDPOINT = "https://models.inference.ai.azure.com"
TOKEN = os.environ.get("GITHUB_TOKEN", "")

# Initialize the client
def get_client():
    if not TOKEN:
        raise ValueError("GITHUB_TOKEN environment variable is not set")
    return ChatCompletionsClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(TOKEN),
    )

async def fetch_available_models() -> List[Dict[str, Any]]:
    # Lookup method and API endpoint lifted from the GitHub Models Extension
    # @link https://github.com/copilot-extensions/github-models-extension 
    """
    Fetch available models from Azure ML API.
    Returns full model information including IDs and metadata.
    """
    try:
        url = "https://api.catalog.azureml.ms/asset-gallery/v1.0/models"
        headers = {"Content-Type": "application/json"}
        body = json.dumps({
            "filters": [
                {"field": "freePlayground", "values": ["true"], "operator": "eq"},
                {"field": "labels", "values": ["latest"], "operator": "eq"}
            ],
            "order": [{"field": "displayName", "direction": "Asc"}]
        })

        response = requests.post(url, headers=headers, data=body)
        if not response.ok:
            raise Exception(f"Failed to fetch models from the model catalog: {response.status_code} {response.text}")

        api_response = response.json()
        models_data = []
        
        # Extract the summaries from the API response
        summaries = api_response.get("summaries", [])
        
        for model in summaries:
            # Extract the model ID from assetId path
            asset_id = model.get("assetId", "")
            model_id = model.get("name", "")
            
            # Map the API response to our model structure
            model_info = {
                "id": model_id,
                "displayName": model.get("displayName", model_id),
                "publisher": model.get("publisher", "Unknown"),
                "summary": model.get("summary", ""),
                "version": model.get("version", ""),
                "context_window": model.get("modelLimits", {}).get("textLimits", {}).get("inputContextWindow", 0),
                "supported_languages": model.get("modelLimits", {}).get("supportedLanguages", []),
                "popularity": model.get("popularity", 0),
                "keywords": model.get("keywords", []),
                "assetId": asset_id
            }
            models_data.append(model_info)
        
        return models_data
    except Exception as e:
        # Fallback to hardcoded list if API fails
        print(f"Warning: Failed to fetch models from API: {str(e)}")
        return [
            {"id": "gpt-4o", "displayName": "GPT-4o", "publisher": "OpenAI", "summary": "OpenAI's most advanced multimodal model", "context_window": 128000},
            {"id": "gpt-4o-mini", "displayName": "GPT-4o-mini", "publisher": "OpenAI", "summary": "Smaller, efficient version of GPT-4o", "context_window": 128000},
            {"id": "Phi-3.5-MoE-instruct", "displayName": "Phi-3.5-MOE Instruct", "publisher": "Microsoft", "summary": "A mixture of experts model from Microsoft", "context_window": 131072},
            {"id": "Phi-3-mini-128k-instruct", "displayName": "Phi-3-Mini Instruct 128k", "publisher": "Microsoft", "summary": "Small model with large context window", "context_window": 131072},
            {"id": "Llama-3.3-70B-Instruct", "displayName": "Meta Llama 3.3 70B Instruct", "publisher": "Meta", "summary": "Advanced reasoning and instruction following", "context_window": 128000},
            {"id": "Meta-Llama-3-8B-Instruct", "displayName": "Meta Llama 3 8B Instruct", "publisher": "Meta", "summary": "Balanced performance and efficiency", "context_window": 8192},
            {"id": "Mistral-large", "displayName": "Mistral Large", "publisher": "Mistral AI", "summary": "Mistral's flagship model for complex reasoning", "context_window": 32768},
            {"id": "Mistral-small", "displayName": "Mistral Small", "publisher": "Mistral AI", "summary": "Efficient model for low-latency use cases", "context_window": 32768}
        ]

# Cache for model data to avoid repeated API calls
_models_cache = None
_models_cache_timestamp = 0
_CACHE_TTL = 600  # 10 minutes in seconds

async def get_models_data() -> List[Dict[str, Any]]:
    """
    Get model data with caching.
    """
    global _models_cache, _models_cache_timestamp
    current_time = asyncio.get_event_loop().time()
    
    # Return cached data if it's still valid
    if _models_cache is not None and (current_time - _models_cache_timestamp) < _CACHE_TTL:
        return _models_cache
    
    # Fetch fresh data
    _models_cache = await fetch_available_models()
    _models_cache_timestamp = current_time
    return _models_cache

@mcp.resource("models://available")
async def get_available_models_resource() -> str:
    """
    Get a list of available models as a formatted resource.
    Includes model ID and display name.
    """
    models = await get_models_data()
    
    # Format the models data into a readable string
    formatted_models = ["# Available Models", ""]
    
    # Group models by publisher
    publishers = {}
    for model in models:
        model_id = model.get("id", "unknown")
        display_name = model.get("displayName", model_id)
        publisher = model.get("publisher", "Unknown")
        context_window = model.get("context_window", 0)
        summary = model.get("summary", "")
        
        if publisher not in publishers:
            publishers[publisher] = []
        
        publishers[publisher].append({
            "id": model_id,
            "display_name": display_name,
            "context_window": context_window,
            "summary": summary
        })
    
    # Sort publishers alphabetically
    for publisher in sorted(publishers.keys()):
        formatted_models.append(f"## {publisher}")
        formatted_models.append("")
        
        # Sort models by display name within each publisher
        for model in sorted(publishers[publisher], key=lambda x: x["display_name"]):
            context_str = f" - {model['context_window']:,} tokens" if model['context_window'] else ""
            summary_str = f" - {model['summary']}" if model['summary'] else ""
            formatted_models.append(f"- **{model['display_name']}** (`{model['id']}`){context_str}{summary_str}")
            
        formatted_models.append("")
    
    return "\n".join(formatted_models)

@mcp.tool()
async def list_available_models(
    include_metadata: bool = False,
    filter_publisher: Optional[str] = None,
    sort_by: str = "displayName"
) -> Dict[str, Any]:
    """
    Get a list of available GitHub models.
    
    Args:
        include_metadata: Whether to include additional model metadata
        filter_publisher: Filter models by publisher (e.g., "OpenAI", "Microsoft")
        sort_by: Sort models by this field (options: "displayName", "publisher", "popularity", "context_window")
    
    Returns:
        Dictionary containing model IDs and optional metadata
    """
    models = await get_models_data()
    
    # Apply publisher filter if specified
    if filter_publisher:
        models = [model for model in models if model.get("publisher", "").lower() == filter_publisher.lower()]
    
    # Apply sorting
    valid_sort_fields = ["displayName", "publisher", "popularity", "context_window", "id"]
    sort_field = sort_by if sort_by in valid_sort_fields else "displayName"
    
    # Default value for missing fields to ensure stable sorting
    default_value = "" if sort_field in ["displayName", "publisher", "id"] else 0
    
    # Sort the models
    sorted_models = sorted(models, key=lambda x: x.get(sort_field, default_value))
    
    if include_metadata:
        return {
            "models": sorted_models,
            "count": len(sorted_models),
            "timestamp": _models_cache_timestamp,
            "publishers": sorted(list(set(model.get("publisher", "Unknown") for model in models))),
            "applied_filters": {
                "publisher": filter_publisher,
                "sort_by": sort_field
            }
        }
    else:
        return {
            "model_ids": [model.get("id") for model in sorted_models],
            "count": len(sorted_models)
        }

@mcp.tool()
async def get_model_details(model_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: The ID of the model to get details for
        
    Returns:
        Dictionary containing model details
    """
    models = await get_models_data()
    
    for model in models:
        if model.get("id") == model_id:
            return {
                "model": model,
                "found": True
            }
    
    return {
        "error": f"Model with ID '{model_id}' not found",
        "found": False,
        "available_models": [model.get("id") for model in models]
    }

@mcp.resource("examples://comparison")
def get_comparison_example() -> str:
    """Example usage of the compare_models tool"""
    example = {
"prompt": "Explain the concept of quantum computing in simple terms.",
        "models": ["gpt-4o", "claude-3-opus-20240229", "Phi-4"],
        "temperature": 0.7
    }
    return json.dumps(example, indent=2)

@mcp.resource("models://usage")
def get_models_usage_guide() -> str:
    """Usage guide for the model comparison tools"""
    return """# GitHub Models Comparison Usage Guide

## Basic Usage

1. First, explore available models:
   - Check the `models://available` resource for a list of all models
   - Use the `list_available_models` tool to get models programmatically

2. Compare models with similar prompts:
   ```json
   {
     "prompt": "Your question or task here",
     "models": ["gpt-4o", "Mistral-Large-2411", "Phi-4"],
     "temperature": 0.7
   }
   ```

3. For batch processing:
   ```json
   {
     "prompts": [
       "What is machine learning?",
       "Explain the difference between supervised and unsupervised learning",
       "How does reinforcement learning work?"
     ],
     "models": ["gpt-4o", "Phi-4"]
   }
   ```

## Advanced Options

- **Filter by publisher**: `list_available_models(filter_publisher="Microsoft")`
- **Include metadata**: `list_available_models(include_metadata=true)`
- **Sort models**: `list_available_models(sort_by="context_window")`
- **Get model details**: `get_model_details(model_id="Phi-4")`
- **Save results**: Use `save_comparison_results` to persist comparisons to a file

## Environment Setup

Ensure your `.env` file contains:
```
GITHUB_TOKEN=your_github_personal_access_token
```
"""

@mcp.tool()
async def compare_models(
    prompt: str, 
    system_message: str = "You are a helpful assistant.",
    models: Optional[List[str]] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1000,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Compare responses from different GitHub models for the same prompt.
    
    Args:
        prompt: The user message to send to all models
        system_message: The system message to use (defaults to "You are a helpful assistant.")
        models: List of model names to compare (defaults to ["gpt-4o", "Mistral-small", "Phi-3-mini-128k-instruct"])
        temperature: Temperature parameter for generation (0.0 to 2.0)
        top_p: Top-p parameter for generation (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        ctx: MCP context object
        
    Returns:
        Dictionary containing responses from each model
    """
    if not models:
        models = ["gpt-4o", "Mistral-small", "Phi-3-mini-128k-instruct"]
    
    # Validate that the requested models exist
    available_models = await get_models_data()
    available_model_ids = [model.get("id") for model in available_models]
    
    # Filter out invalid models and warn about them
    valid_models = []
    for model_name in models:
        if model_name in available_model_ids:
            valid_models.append(model_name)
        elif ctx:
            ctx.info(f"Warning: Model '{model_name}' not found in available models. Skipping.")
    
    if not valid_models:
        return {
            "error": "No valid models specified",
            "available_models": available_model_ids
        }
    
    client = get_client()
    results = {}
    total_models = len(valid_models)
    
    for idx, model_name in enumerate(valid_models):
        if ctx:
            ctx.info(f"Processing model {idx+1}/{total_models}: {model_name}")
            await ctx.report_progress(idx, total_models)
        
        try:
            response = client.complete(
                messages=[
                    SystemMessage(system_message),
                    UserMessage(prompt),
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                model=model_name
            )
            
            results[model_name] = {
                "content": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            results[model_name] = {"error": str(e)}
    
    if ctx:
        await ctx.report_progress(total_models, total_models)
        ctx.info("Model comparison completed")
    
    # Return formatted results
    return {
        "results": results,
        "summary": {
            "models_compared": valid_models,
            "prompt": prompt,
            "system_message": system_message
        }
    }

@mcp.prompt()
def compare_models_prompt(prompt: str, models: str = "") -> str:
    """Create a comparison prompt with selected models"""
    if models:
        return f"""Please compare how different AI models respond to this prompt: 
        
Prompt: {prompt}

Please use these specific models: {models}"""
    else:
        return f"""Please compare how different AI models respond to this prompt: 
        
Prompt: {prompt}

Please use the default models available."""

@mcp.prompt()
def evaluate_responses_prompt(prompt: str, models: str) -> str:
    """Evaluate differences between model responses"""
    return f"""I've compared different AI models' responses to this prompt:

Prompt: {prompt}

Models used: {models}

Can you analyze the differences between their responses? Please highlight:
1. Key differences in content and approach
2. Strengths and weaknesses of each response
3. Which model performed best for this specific prompt and why"""

@mcp.tool()
async def batch_compare(
    prompts: List[str],
    system_message: str = "You are a helpful assistant.",
    models: Optional[List[str]] = None,
    temperature: float = 1.0,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Compare model responses across multiple prompts.
    
    Args:
        prompts: List of prompts to test
        system_message: System message to use
        models: List of models to compare
        temperature: Temperature setting
        ctx: MCP context
        
    Returns:
        Dictionary with results for each prompt and model
    """
    if not models:
        # Use first two models from available models as defaults
        available_models = await get_models_data()
        if len(available_models) >= 2:
            models = [available_models[0].get("id"), available_models[1].get("id")]
        else:
            models = ["gpt-4o-mini", "mistral-small"]
    
    results = {}
    total = len(prompts)
    
    for idx, prompt in enumerate(prompts):
        if ctx:
            ctx.info(f"Processing prompt {idx+1}/{total}: {prompt[:50]}...")
            await ctx.report_progress(idx, total)
        
        prompt_result = await compare_models(
            prompt=prompt,
            system_message=system_message,
            models=models,
            temperature=temperature,
            ctx=ctx
        )
        
        results[f"prompt_{idx+1}"] = {
            "prompt": prompt,
            "results": prompt_result["results"]
        }
    
    if ctx:
        await ctx.report_progress(total, total)
    
    return {
        "batch_results": results,
        "summary": {
            "prompts_processed": len(prompts),
            "models_compared": models
        }
    }

@mcp.tool()
async def save_comparison_results(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    models: Optional[List[str]] = None,
    output_path: str = "./model_comparison_results.json",
    ctx: Context = None
) -> str:
    """
    Run model comparison and save results to a file.
    
    Args:
        prompt: Prompt to test
        system_message: System message
        models: Models to compare
        output_path: File path to save results
        ctx: MCP context
        
    Returns:
        Path to saved results file
    """
    results = await compare_models(
        prompt=prompt,
        system_message=system_message,
        models=models,
        ctx=ctx
    )
    
    # Save results to file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return f"Results saved to {output_path}"

if __name__ == "__main__":
    mcp.run()