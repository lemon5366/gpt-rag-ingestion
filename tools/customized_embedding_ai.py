import logging
import os

import requests
from typing import Optional, List


EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "DEFAULT_VALUE")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")


def call_embedding_api_single(text: str) -> Optional[List[float]]:
    """单个文本的embedding请求 - 避免批量请求的内存问题"""
    try:
        logging.info(f"[call_embedding_api_single]Getting embedding for text length: {len(text)}")
        r = requests.post(
            EMBEDDING_ENDPOINT,
            json={"input": text, "model": EMBEDDING_MODEL},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        
        # 支持两种响应格式
        if isinstance(data, dict) and "data" in data and data["data"]:
            embedding = data["data"][0].get("embedding")
            if isinstance(embedding, list):
                return embedding
                
        if isinstance(data, dict) and "embedding" in data:
            embedding = data["embedding"]
            if isinstance(embedding, list):
                return embedding
                
        logging.error(f"[call_embedding_api_single]Unexpected embedding response format")
        return None
    except Exception as e:
        logging.error(f"[call_embedding_api_single]Embedding request failed: {e}")
        return None