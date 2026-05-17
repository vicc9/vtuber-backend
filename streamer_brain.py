import asyncio
import json
import os
from dotenv import load_dotenv
from groq import AsyncGroq
from memory_manager import MemoryManager
from web_search import WebSearcher

load_dotenv()

# Function Calling 的工具定義
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "當觀眾詢問即時資訊、新聞、時事、天氣、股價、"
                "或任何你不確定的事實性問題時，呼叫此工具搜尋網路。"
                "觀眾說「查一下」、「搜尋」、「最新」等關鍵字時也應呼叫。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜尋的關鍵字，用繁體中文或英文皆可"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

class StreamerBrain:
    def __init__(self):
        print("🧠 初始化主播大腦中 (使用模型: Groq + llama3)...")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ 找不到 GROQ_API_KEY，請檢查 .env 檔案！")

        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.history = []

        # 記憶系統
        self.memory = MemoryManager()
        self.long_term_memory = self.memory.load_recent_summary(limit=3)
        if self.long_term_memory:
            print(f"📚 已載入長期記憶:\n{self.long_term_memory}")
        else:
            print("📭 尚無長期記憶，這是全新開始")

        # 網路搜尋
        self.searcher = WebSearcher()

        print("✅ 主播大腦初始化完成！")

    def _build_system_prompt(self) -> str:
        base = """你是知性、幽默且具備個人特色的即時 VTuber 主播小光。
請做出生動、自然的口語化回應。
當你使用網路搜尋結果時，請自然地融入對話，不要直接唸出搜尋結果。
嚴格以 JSON 物件輸出，絕對不能包含任何其他文字或 markdown，必須包含以下三個欄位：
{
  "dialogue": "你的口語播報稿文字",
  "emotion": "從 [joy, anger, sadness, neutral] 中選一個",
  "action": "從 [Wave, Nod, Think, 無動作] 中選一個"
}"""

        if self.long_term_memory:
            base += f"""

## 你的長期記憶（過去直播累積的重要資訊）
{self.long_term_memory}

請自然地運用這些記憶，例如主動提起之前聊過的話題，或記得熟悉觀眾的名字。"""

        return base

    async def generate_live_response(
        self,
        user_input: str,
        rag_context: str = "無特別背景資訊",
        speaker_name: str = "觀眾"
    ) -> dict:

        # 儲存觀眾訊息
        self.memory.save_message("user", user_input, speaker_name=speaker_name)

        user_prompt = f"""當前觀眾對你說的話: "{user_input}"
你大腦中的背景記憶資訊: "{rag_context}"
請以 JSON 格式回應。"""

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            *self.history,
            {"role": "user", "content": user_prompt}
        ]

        try:
            # ── 第一輪：讓 LLM 決定要不要搜尋 ──
            first_response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=512,
            )

            choice = first_response.choices[0]
            search_context = ""
            text = ""

            # ── 如果 LLM 決定要搜尋 ──
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                try:
                    for tool_call in choice.message.tool_calls:
                        if tool_call.function.name == "web_search":
                            args = json.loads(tool_call.function.arguments)
                            query = args.get("query", user_input)
                            search_result = self.searcher.search(query)
                            search_context = search_result
                            print(f"🔍 搜尋結果已取得，長度: {len(search_result)} 字")

                    messages.append(choice.message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": choice.message.tool_calls[0].id,
                        "content": search_context
                    })

                    second_response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=512,
                        response_format={"type": "json_object"}
                    )
                    text = second_response.choices[0].message.content.strip()

                except Exception as search_err:
                    # ── Tool call 格式錯誤時，降級為直接回答 ──
                    print(f"⚠ Function calling 失敗，改用直接回答: {search_err}")
                    text = ""  # 讓下面的 fallback 重新呼叫

            # ── 不需要搜尋，或搜尋失敗降級 ──
            if not text:
                final_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=512,
                    response_format={"type": "json_object"}
                )
                text = final_response.choices[0].message.content.strip()

            data = json.loads(text)

            # 更新對話歷史
            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": text})
            if len(self.history) > 20:
                self.history = self.history[-20:]

            # 儲存 AI 回應
            self.memory.save_message(
                "assistant",
                data.get("dialogue", ""),
                emotion=data.get("emotion", "neutral")
            )

            return data

        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析錯誤: {e}")
            fallback = {
                "dialogue": "不好意思，我剛剛思緒有點亂，可以再說一次嗎？",
                "emotion": "neutral",
                "action": "Think"
            }
            self.memory.save_message("assistant", fallback["dialogue"])
            return fallback

        except Exception as e:
            print(f"❌ 大腦生成錯誤: {e}")
            fallback = {
                "dialogue": "糟糕，我的大腦剛剛當機了一下，請再說一次好嗎？",
                "emotion": "sadness",
                "action": "Think"
            }
            self.memory.save_message("assistant", fallback["dialogue"])
            return fallback

    async def generate_session_summary(self) -> str:
        """場次結束時，用 LLM 自動生成摘要並儲存"""
        conversations = self.memory.load_session_conversations()
        if not conversations:
            print("📭 本場無對話紀錄，略過摘要生成")
            return ""

        conv_text = "\n".join([
            f"{'觀眾' + row['speaker_name'] if row['role'] == 'user' else '小光'}: {row['content']}"
            for row in conversations
        ])

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": f"""請用繁體中文分析以下直播對話，以 JSON 格式回傳：
{{
  "summary": "100字以內的對話摘要",
  "key_facts": {{
    "喜歡": ["觀眾提到喜歡的東西"],
    "常來觀眾": ["活躍的觀眾名稱"],
    "聊過的話題": ["主要話題"]
  }}
}}

對話內容：
{conv_text}"""
                }],
                response_format={"type": "json_object"},
                max_tokens=300,
            )

            result = json.loads(response.choices[0].message.content)
            summary = result.get("summary", "")
            key_facts = result.get("key_facts", {})
            self.memory.save_session_summary(summary=summary, key_facts=key_facts)
            return summary

        except Exception as e:
            print(f"⚠ 摘要生成失敗: {e}")
            return ""


async def main():
    brain = StreamerBrain()

    print("\n測試一：一般對話（不需要搜尋）")
    res1 = await brain.generate_live_response("嗨！主播今天過得好嗎？")
    print(json.dumps(res1, indent=4, ensure_ascii=False))

    print("\n測試二：需要搜尋（AI 自動判斷）")
    res2 = await brain.generate_live_response("今天台灣有什麼新聞？")
    print(json.dumps(res2, indent=4, ensure_ascii=False))

    print("\n測試三：觀眾強制觸發搜尋")
    res3 = await brain.generate_live_response("查一下今天台北天氣")
    print(json.dumps(res3, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())