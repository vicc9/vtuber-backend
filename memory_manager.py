import os
import uuid
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class MemoryManager:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        self.supabase: Client = create_client(url, key)
        self.session_id = str(uuid.uuid4())
        print(f"🧠 記憶系統啟動，場次 ID: {self.session_id}")

    def save_message(self, role: str, content: str,
                     speaker_name: str = "anonymous", emotion: str = "neutral"):
        """儲存一輪對話到 Supabase"""
        try:
            self.supabase.table("conversations").insert({
                "session_id": self.session_id,
                "role": role,
                "content": content,
                "speaker_name": speaker_name,
                "emotion": emotion,
            }).execute()
        except Exception as e:
            print(f"⚠ 儲存對話失敗: {e}")

    def load_recent_summary(self, limit: int = 3) -> str:
        """載入最近幾場的摘要，注入 system prompt 用"""
        try:
            result = self.supabase.table("memory_summary") \
                .select("summary, key_facts, created_at") \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()

            if not result.data:
                return ""

            summaries = []
            for row in reversed(result.data):
                date = row["created_at"][:10]
                summaries.append(f"[{date}] {row['summary']}")
                facts = row.get("key_facts") or {}
                if facts.get("喜歡"):
                    summaries.append(f"  → 喜歡：{', '.join(facts['喜歡'])}")
                if facts.get("常來觀眾"):
                    summaries.append(f"  → 熟悉觀眾：{', '.join(facts['常來觀眾'])}")
                if facts.get("聊過的話題"):
                    summaries.append(f"  → 聊過：{', '.join(facts['聊過的話題'])}")

            return "\n".join(summaries)

        except Exception as e:
            print(f"⚠ 載入長期記憶失敗: {e}")
            return ""

    def load_session_conversations(self, limit: int = 50) -> list:
        """載入本場所有對話，供場次結束時生成摘要用"""
        try:
            result = self.supabase.table("conversations") \
                .select("role, content, speaker_name, emotion") \
                .eq("session_id", self.session_id) \
                .order("created_at") \
                .limit(limit) \
                .execute()
            return result.data or []
        except Exception as e:
            print(f"⚠ 載入本場對話失敗: {e}")
            return []

    def save_session_summary(self, summary: str, key_facts: dict = None):
        """儲存場次摘要"""
        try:
            self.supabase.table("memory_summary").insert({
                "session_id": self.session_id,
                "summary": summary,
                "key_facts": key_facts or {},
            }).execute()
            print(f"✅ 場次摘要已儲存：{summary}")
        except Exception as e:
            print(f"⚠ 摘要儲存失敗: {e}")
