# supabase_rag.py
# ══════════════════════════════════════════════════════════════
# 架構說明：
#   Embedding 在 Render 本地用 sentence-transformers 計算（384維）
#   → 向量 + 文字直接透過 supabase-py REST API 存入 Supabase
#   → 查詢時呼叫 match_memories RPC
#
# 這樣完全不依賴 Supabase 的 Embedding API，
# 解決 Render 免費方案 [Errno -5] DNS 失敗問題。
# ══════════════════════════════════════════════════════════════

import os
import asyncio
from typing import Optional
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# 模型與 Supabase 初始化
# ─────────────────────────────────────────
_model: Optional[SentenceTransformer] = None
_supabase: Optional[Client] = None

def _get_model() -> SentenceTransformer:
    """延遲載入 Embedding 模型（只載入一次）"""
    global _model
    if _model is None:
        print("📦 載入本地 Embedding 模型 (paraphrase-multilingual-MiniLM-L12-v2)...")
        _model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        print("✅ Embedding 模型載入完成（384維）")
    return _model

def _get_supabase() -> Client:
    """建立 Supabase client（只建立一次）"""
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        # ✅ 優先使用 service_role key 以繞過 RLS；若無則用 anon key
        key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("❌ 找不到 SUPABASE_URL 或 SUPABASE_SERVICE_KEY/SUPABASE_KEY 環境變數")
        _supabase = create_client(url, key)
        print("✅ Supabase client 建立完成")
    return _supabase


# ─────────────────────────────────────────
# StreamerRAG 主類別
# ─────────────────────────────────────────
class StreamerRAG:

    def __init__(self):
        print("🗄️ 初始化 Supabase RAG（本地 Embedding 模式）...")
        # 啟動時預載模型，避免第一次查詢時延遲
        try:
            _get_model()
            _get_supabase()
        except Exception as e:
            print(f"⚠️ RAG 初始化警告: {e}")

    def _embed(self, text: str) -> list[float]:
        """將文字轉為 384 維向量（同步，在 Render 本地執行）"""
        model = _get_model()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    async def store_memory(self, content: str) -> bool:
        """
        將文字內容向量化並存入 Supabase memories 表。
        在 executor 中執行同步 Embedding 計算，不阻塞 event loop。
        """
        if not content or not content.strip():
            return False

        try:
            loop = asyncio.get_event_loop()
            # Embedding 計算是 CPU 密集，放到 executor
            embedding = await loop.run_in_executor(None, self._embed, content)

            supabase = _get_supabase()
            result = supabase.table("memories").insert({
                "content":   content,
                "embedding": embedding,   # list[float] → supabase-py 自動序列化
            }).execute()

            if result.data:
                print(f"💾 記憶已存入 Supabase（id={result.data[0].get('id')}）")
                return True
            else:
                print(f"⚠️ 儲存記憶失敗（無回傳資料）: {result}")
                return False

        except Exception as e:
            print(f"⚠️ 儲存記憶失敗: {e}")
            return False

    async def retrieve_memory(self, query: str, k: int = 2, threshold: float = 0.3) -> str:
        """
        向量化查詢文字，呼叫 Supabase match_memories RPC 取得相似記憶。
        threshold 降低至 0.3，避免中文短句相似度偏低時查不到結果。
        """
        print(f"🧠 正在 Supabase 搜尋關於「{query}」的相關記憶...")
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, self._embed, query)

            supabase = _get_supabase()
            result = supabase.rpc("match_memories", {
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
        """
        將長文本切片後批次存入 Supabase。
        （保留與舊版 news_rag.py 相同功能）
        """
        if not text or not text.strip():
            print("⚠️ 沒有內容可以寫入！")
            return False

        # 簡單按句號/換行切片
        chunks = []
        current = ""
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

        print(f"✂️ 切分為 {len(chunks)} 個記憶節點，開始存入...")
        success = 0
        for chunk in chunks:
            if await self.store_memory(chunk):
                success += 1

        print(f"✅ 成功存入 {success}/{len(chunks)} 個記憶節點")
        return success > 0