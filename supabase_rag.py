# supabase_rag.py
# ══════════════════════════════════════════════════════════════
# 架構：用 Groq 的 embedding API 取代本地 sentence-transformers
#       → 零額外記憶體，解決 Render 免費方案 512MB OOM 問題
#       → 仍使用你的 Supabase memories 表（384維 pgvector）
#
# Groq embedding 模型：nomic-embed-text-v1.5（768維）
# ⚠️  注意：Groq embedding 是 768 維，需更新 Supabase schema
#     執行附帶的 fix_embedding_dim.sql 更新欄位維度
# ══════════════════════════════════════════════════════════════

import os
import asyncio
import httpx
from typing import Optional
from supabase import create_client, Client

_supabase: Optional[Client] = None

HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN")

def _get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("缺少 SUPABASE_URL 或 SUPABASE_SERVICE_KEY")
        _supabase = create_client(url, key)
    return _supabase


async def _embed(text: str) -> list[float]:
    """呼叫 Hugging Face API 生成 384 維度向量"""
    if not text or not text.strip():
        return []
        
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
        
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.post(
                HF_EMBED_URL,
                headers=headers,
                json={"inputs": text}
            )
            response.raise_for_status()
            data = response.json()
            
            # 處理 API 可能回傳的模型載入中狀態
            if isinstance(data, dict) and "error" in data:
                print(f"⚠️ HF 模型載入中或發生錯誤: {data['error']}")
                return []
                
            # API 回傳可能是 list[float] 或 list[list[float]]
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], list):
                    return data[0]
                return data
            return []
            
        except Exception as e:
            print(f"❌ HF Embedding 失敗: {e}")
            return []


class StreamerRAG:

    def __init__(self):
        print(f"🗄️ 初始化 Supabase RAG（Groq Embedding 模式，{EMBED_DIM}維）...")
        try:
            _get_supabase()
            print("✅ Supabase RAG 初始化完成")
        except Exception as e:
            print(f"⚠️ RAG 初始化警告: {e}")

    async def store_memory(self, content: str) -> bool:
        """向量化並存入 Supabase memories 表"""
        if not content or not content.strip():
            return False
        try:
            embedding = await _embed(content)
            result = _get_supabase().table("memories").insert({
                "content":   content,
                "embedding": embedding,
            }).execute()
            if result.data:
                print(f"💾 記憶存入成功（id={result.data[0].get('id')}）")
                return True
            return False
        except Exception as e:
            print(f"⚠️ 儲存記憶失敗: {e}")
            return False

    async def retrieve_memory(self, query: str, k: int = 2, threshold: float = 0.3) -> str:
        """向量化查詢，呼叫 match_memories RPC"""
        print(f"🧠 正在搜尋關於「{query}」的相關記憶...")
        try:
            embedding = await _embed(query)
            result = _get_supabase().rpc("match_memories", {
                "query_embedding": embedding,
                "match_threshold":  threshold,
                "match_count":      k,
            }).execute()

            if not result.data:
                return "大腦中沒有找到相關記憶。"

            memories = [row["content"] for row in result.data]
            print(f"✅ 找到 {len(memories)} 筆相關記憶")
            return "\n".join(memories)

        except Exception as e:
            print(f"⚠️ 記憶檢索失敗: {e}")
            return "大腦中沒有找到相關記憶。"

    async def ingest_text(self, text: str, chunk_size: int = 300) -> bool:
        """長文本切片後批次存入"""
        if not text or not text.strip():
            return False
        chunks, current = [], ""
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            current += line + " "
            if len(current) >= chunk_size:
                chunks.append(current.strip())
                current = ""
        if current.strip():
            chunks.append(current.strip())

        print(f"✂️ 切分為 {len(chunks)} 個節點，開始存入...")
        success = sum(1 for c in chunks if await self.store_memory(c))
        print(f"✅ 成功存入 {success}/{len(chunks)}")
        return success > 0