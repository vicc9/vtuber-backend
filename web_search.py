import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

class WebSearcher:
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        # ✅ 修正：API key 不存在時不崩潰，search() 呼叫時才回報
        if not api_key:
            print("⚠️ 找不到 TAVILY_API_KEY，網路搜尋功能將停用")
            self.client = None
        else:
            try:
                self.client = TavilyClient(api_key=api_key)
                print("🔍 網路搜尋模組初始化完成！")
            except Exception as e:
                print(f"⚠️ Tavily 初始化失敗: {e}")
                self.client = None

    def search(self, query: str, max_results: int = 3) -> str:
        """執行搜尋，回傳整理好的文字摘要"""
        if self.client is None:
            print("⚠️ Tavily client 未初始化，跳過搜尋")
            return "網路搜尋功能未啟用"
        try:
            print(f"🌐 正在搜尋：{query}")
            response = self.client.search(
                query=query + " 繁體中文",
                search_depth="basic",
                max_results=max_results,
                include_answer=True,       # 讓 Tavily 直接給一個整合答案
            )

            # 優先用 Tavily 的整合答案
            if response.get("answer"):
                result = response["answer"]
            else:
                # 否則拼接各來源的摘要
                parts = []
                for r in response.get("results", []):
                    title = r.get("title", "")
                    content = r.get("content", "")[:200]  # 限制長度
                    parts.append(f"【{title}】{content}")
                result = "\n".join(parts)

            print(f"✅ 搜尋完成，取得 {len(response.get('results', []))} 筆結果")
            return result if result else "搜尋未找到相關結果"

        except Exception as e:
            print(f"❌ 搜尋失敗: {e}")
            return "網路搜尋目前無法使用"