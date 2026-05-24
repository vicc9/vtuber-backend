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

GROQ_EMBED_URL = "https://api.groq.com/openai/v1/embeddings"
EMBED_MODEL    = "nomic-embed-text-v1.5"
EMBED_DIM      = 768   # nomic-embed-text-v1.5 輸出維度

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
    """
    呼叫 Groq Embedding API，回傳向量。
    完全走外部 API，不佔用本地記憶體。
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 GROQ_API_KEY")

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            GROQ_EMBED_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model": EMBED_MODEL,
                "input": text,
            }
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API 錯誤 [{resp.status_code}]: {resp.text}")

        data = resp.json()
        return data["data"][0]["embedding"]


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