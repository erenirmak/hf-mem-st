"""
Memory estimation utilities for HuggingFace models.
Extracts the core logic from hf_mem to be used programmatically.
"""

import asyncio
import json
import math
import os
import struct
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple

import httpx

MAX_METADATA_SIZE = 100_000
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 10.0))
MAX_CONCURRENCY = int(os.getenv("MAX_WORKERS", min(32, (os.cpu_count() or 1) + 4)))


async def get_json_file(
    client: httpx.AsyncClient, url: str, headers: Optional[Dict[str, str]] = None
) -> Any:
    """Fetch a JSON file from URL."""
    response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


async def fetch_safetensors_metadata(
    client: httpx.AsyncClient, url: str, headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Fetch safetensors metadata from URL."""
    headers = {"Range": f"bytes=0-{MAX_METADATA_SIZE}", **(headers or {})}
    response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    metadata = response.read()
    metadata_size = struct.unpack("<Q", metadata[:8])[0]

    if metadata_size < MAX_METADATA_SIZE:
        metadata = metadata[8 : metadata_size + 8]
        return json.loads(metadata)

    metadata = metadata[8 : MAX_METADATA_SIZE + 8]
    headers["Range"] = f"bytes={MAX_METADATA_SIZE + 1}-{metadata_size + 7}"

    response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    metadata += response.read()
    return json.loads(metadata)


def _bytes_to_gb(nbytes: int) -> float:
    """Convert bytes to gigabytes."""
    return nbytes / (1024**3)


def _format_number(n: int) -> str:
    """Format large numbers with K, M, B, T suffixes."""
    n = float(n)
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000.0:
            return f"{int(n)}" if unit == "" else f"{n:.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}P"


def calculate_memory_requirements_transformers(
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate memory requirements for transformer models from safetensors metadata.
    
    Returns a dict with:
    - total_gb: Total memory in GB
    - total_params: Total parameters
    - breakdown: Dict of dtype -> (params, bytes, gb)
    """
    ppdt: Dict[str, Tuple[int, int]] = {}
    
    for key, value in metadata.items():
        if key in {"__metadata__"}:
            continue
        
        if value["dtype"] not in ppdt:
            ppdt[value["dtype"]] = (0, 0)

        dtype = value["dtype"]
        match dtype:
            case "F64" | "I64" | "U64":
                dtype_b = 8
            case "F32" | "I32" | "U32":
                dtype_b = 4
            case "F16" | "BF16" | "I16" | "U16":
                dtype_b = 2
            case "F8_E5M2" | "F8_E4M3" | "I8" | "U8":
                dtype_b = 1
            case _:
                raise RuntimeError(f"DTYPE={dtype} NOT HANDLED")

        current_shape = math.prod(value["shape"])
        current_shape_bytes = current_shape * dtype_b

        ppdt[dtype] = (
            ppdt[dtype][0] + current_shape,
            ppdt[dtype][1] + current_shape_bytes,
        )

    total_bytes = sum(nbytes for _, nbytes in ppdt.values())
    total_params = sum(params for params, _ in ppdt.values())
    total_gb = _bytes_to_gb(total_bytes)

    breakdown = {
        dtype: {
            "params": params,
            "bytes": nbytes,
            "gb": _bytes_to_gb(nbytes),
            "params_formatted": _format_number(params),
            "percentage": (nbytes / total_bytes * 100) if total_bytes > 0 else 0,
        }
        for dtype, (params, nbytes) in ppdt.items()
    }

    return {
        "total_gb": total_gb,
        "total_params": total_params,
        "total_params_formatted": _format_number(total_params),
        "total_bytes": total_bytes,
        "breakdown": breakdown,
    }


async def estimate_model_memory(
    model_id: str, revision: str = "main"
) -> Dict[str, Any]:
    """
    Estimate memory requirements for a HuggingFace model.
    
    Returns a dict with memory breakdown by dtype.
    """
    headers = {}
    if token := os.getenv("HF_TOKEN", None):
        headers["Authorization"] = f"Bearer {token}"

    client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_keepalive_connections=MAX_CONCURRENCY,
            max_connections=MAX_CONCURRENCY,
        ),
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        http2=True,
        follow_redirects=True,
    )

    try:
        url = f"https://huggingface.co/api/models/{model_id}/tree/{revision}?recursive=true"
        files = await get_json_file(client=client, url=url, headers=headers)

        file_paths = [
            f["path"]
            for f in files
            if f.get("path", None) is not None and f.get("type", None) == "file"
        ]

        if "model.safetensors" in file_paths:
            url = f"https://huggingface.co/{model_id}/resolve/{revision}/model.safetensors"
            metadata = await fetch_safetensors_metadata(
                client=client, url=url, headers=headers
            )
            result = calculate_memory_requirements_transformers(metadata)
            result["model_type"] = "transformer"
            return result
            
        elif "model.safetensors.index.json" in file_paths:
            url = f"https://huggingface.co/{model_id}/resolve/{revision}/model.safetensors.index.json"
            files_index = await get_json_file(client=client, url=url, headers=headers)

            urls = {
                f"https://huggingface.co/{model_id}/resolve/{revision}/{f}"
                for f in set(files_index["weight_map"].values())
            }

            semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

            async def fetch_semaphore(url: str) -> Dict[str, Any]:
                async with semaphore:
                    return await fetch_safetensors_metadata(
                        client=client, url=url, headers=headers
                    )

            tasks = [asyncio.create_task(fetch_semaphore(url)) for url in urls]
            metadata_list: List[Dict[str, Any]] = await asyncio.gather(
                *tasks, return_exceptions=False
            )

            metadata = reduce(lambda acc, metadata: acc | metadata, metadata_list, {})
            result = calculate_memory_requirements_transformers(metadata)
            result["model_type"] = "transformer_sharded"
            return result
        
        else:
            return {
                "error": "Model format not supported. Only transformer and diffuser models are currently supported."
            }
    finally:
        await client.aclose()


async def search_huggingface_models(query: str, limit: int = 10) -> List[str]:
    """
    Search for models on HuggingFace Hub matching the query.
    Returns a list of model IDs.
    """
    headers = {}
    if token := os.getenv("HF_TOKEN", None):
        headers["Authorization"] = f"Bearer {token}"
    
    client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT), http2=True)
    
    try:
        # Search using HuggingFace API
        url = f"https://huggingface.co/api/models?search={query}&limit={limit}&sort=downloads&direction=-1"
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        
        models = response.json()
        if isinstance(models, list):
            return [model.get('id', '') for model in models if model.get('id')]
        return []
    except Exception as e:
        print(f"Error searching models: {e}")
        return []
    finally:
        await client.aclose()
