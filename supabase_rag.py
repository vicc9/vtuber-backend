import os
import httpx
from supabase import create_client, Client

class StreamerRAG:
    def __init__(self):
        # 初始化 Supabase
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            print("⚠️ 警告：未設定 SUPABASE_URL 或 SUPABASE_KEY")
        self.supabase: Client = create_client(url, key)
        
        # Hugging Face 免費的 Embedding 模型 API (不用 Token 也可以小量使用)
        self.embedding_api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

    async def _get_embedding(self, text: str) -> list[float]:
        """呼叫外部 API 將文字轉成 384 維度的向量，避免吃光 Render 伺服器記憶體"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self.embedding_api_url, json={"inputs": text})
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"❌ 向量化失敗: {response.status_code} - {response.text}")
                    return [0.0] * 384
        except Exception as e:
            print(f"❌ Embedding API 發生錯誤: {e}")
            return [0.0] * 384

    async def add_memory(self, text: str):
        """將新記憶存入 Supabase"""
        embedding = await self._get_embedding(text)
        if any(embedding):
            self.supabase.table("memories").insert({
                "content": text,
                "embedding": embedding
            }).execute()
            print(f"💾 已將記憶存入 Supabase: {text}")

    async def retrieve_memory(self, query: str, k: int = 2) -> str:
        """從 Supabase 搜尋相關的歷史記憶"""
        try:
            query_embedding = await self._get_embedding(query)
            if not any(query_embedding):
                return ""

            # 呼叫我們在第一步建立的 SQL 函數 (RPC)
            response = self.supabase.rpc(
                'match_memories',
                {
                    'query_embedding': query_embedding, 
                    'match_threshold': 0.3, # 相似度門檻，可根據需求調整
                    'match_count': k
                }
            ).execute()

            if response.data:
                context = "\n".join([f"- {item['content']}" for item in response.data])
                print(f"🔍 從 Supabase 撈到相關記憶:\n{context}")
                return context
            return ""
            
        except Exception as e:
            print(f"❌ Supabase 搜尋失敗: {e}")
            return ""