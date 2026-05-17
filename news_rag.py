# news_rag.py
import os
import requests
from bs4 import BeautifulSoup

# 匯入 LangChain 向量庫與切片工具
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class StreamerRAG:
    def __init__(self, db_path="./chroma_db"):
        print("🗄️ 初始化 ChromaDB 向量記憶庫與嵌入模型 (nomic-embed-text)...")
        # 使用我們剛剛下載的 nomic-embed-text 作為文字轉向量的引擎
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.db_path = db_path
        self.collection_name = "streamer_knowledge"
        
        # 初始化/載入本地持久化向量資料庫
        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path
        )

    def ingest_text_data(self, text_content, source_name="custom_input"):
        """將純文本切片並寫入資料庫"""
        if not text_content.strip():
            print("⚠️ 沒有內容可以寫入！")
            return False

        print(f"✂️ 正在將文本進行記憶切片 ({source_name})...")
        # 設定切片大小為 300 字，前後重疊 50 字保持上下文語意連貫
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_text(text_content)

        if chunks:
            print(f"💾 寫入 {len(chunks)} 個記憶節點到本地資料庫...")
            # 附帶來源詮釋資料 (Metadata)，讓主播知道這句話是從哪看來的
            metadatas = [{"source": source_name} for _ in chunks]
            self.db.add_texts(texts=chunks, metadatas=metadatas)
            print("✅ 記憶存檔成功！")
            return True
        return False

    def crawl_and_ingest_url(self, url):
        """爬取網頁文字並直接存入記憶"""
        try:
            print(f"🌐 正在前往抓取資訊: {url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            
            # 確保中文編碼解析正確
            res.encoding = res.apparent_encoding
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # 提取所有段落文字 (過濾掉太短的導覽列雜訊)
            paragraphs = [p.text.strip() for p in soup.find_all('p') if len(p.text.strip()) > 20]
            full_text = "\n".join(paragraphs)
            
            if not full_text:
                print("⚠️ 找不到足夠的內文段落，請確認該網頁結構。")
                return False
                
            return self.ingest_text_data(full_text, source_name=url)
            
        except Exception as e:
            print(f"❌ 網頁抓取失敗: {e}")
            return False

    def retrieve_memory(self, query, k=2):
        """根據觀眾的提問，回索最相關的 k 個記憶片段"""
        print(f"🧠 正在大腦深處搜尋關於「{query}」的相關資訊...")
        results = self.db.similarity_search(query, k=k)
        
        if not results:
            return "大腦中沒有找到相關記憶。"
            
        # 將找到的記憶組合起來，準備提供給大腦 (StreamerBrain) 參考
        combined_context = "\n".join([f"情報來源({doc.metadata.get('source', '未知')}): {doc.page_content}" for doc in results])
        return combined_context

# =========================================================
# 本地端直接測試記憶系統運作
# =========================================================
if __name__ == "__main__":
    rag = StreamerRAG()
    
    print("\n--- 測試一：手動輸入專屬人設與快訊 ---")
    sample_news = "2026年最新科技與人設快訊：本頻道的主播叫做『小光』，最喜歡喝珍珠奶茶。近期虛擬實境結合生成式 AI 成為主流，VTuber 開始具備自主意識與長期記憶。"
    rag.ingest_text_data(sample_news, source_name="主播設定集")
    
    print("\n--- 測試二：測試大腦檢索能力 ---")
    # 測試問一個剛寫入的問題
    answer = rag.retrieve_memory("請問主播叫什麼名字？喜歡什麼？")
    
    print("\n💡 檢索到的原始記憶結果：")
    print(answer)