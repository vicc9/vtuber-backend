import os
import asyncio
import httpx
from typing import Optional
from supabase import create_client, Client

_supabase: Optional[Client] = None

# ✅ 改用 Hugging Face 免費 API，不佔用本機 RAM，且完美對應 Supabase 預設的 384 維度！
HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN")

def _get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("缺少 SUPABASE_URL 或 SUPABASE_KEY 環境變數")
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
        # ✅ 修正：移除錯誤的變數，改為正確的顯示訊息
        print("🗄️ 初始化 Supabase RAG（Hugging Face Embedding 模式，384維）...")

    async def retrieve_memory(self, query: str, k: int = 2, threshold: float = 0.3) -> str:
        """向量化查詢，呼叫 match_memories RPC"""
        print(f"🧠 正在搜尋關於「{query}」的相關記憶...")
        try:
            embedding = await _embed(query)
            if not embedding:
                return "大腦中沒有找到相關記憶。"

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
        if current:
            chunks.append(current.strip())

        success_count = 0
        for chunk in chunks:
            try:
                embedding = await _embed(chunk)
                if embedding:
                    _get_supabase().table("memories").insert({
                        "content": chunk,
                        "embedding": embedding
                    }).execute()
                    success_count += 1
            except Exception as e:
                print(f"⚠️ 寫入記憶失敗: {e}")
                
        print(f"✅ 成功存入 {success_count}/{len(chunks)} 筆記憶")
        return success_count > 0