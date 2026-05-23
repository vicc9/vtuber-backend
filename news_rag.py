import os
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

class StreamerRAG:
    def __init__(self, db_path="./chroma_db"):
        print("🗄️ 初始化 ChromaDB 向量記憶庫與嵌入模型...")
        # ✅ 改用 HuggingFace 免費嵌入模型，不需要 Ollama
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.db_path = db_path
        self.collection_name = "streamer_knowledge"

        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path
        )

    def ingest_text_data(self, text_content, source_name="custom_input"):
        if not text_content.strip():
            print("⚠️ 沒有內容可以寫入！")
            return False

        print(f"✂️ 正在將文本進行記憶切片 ({source_name})...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_text(text_content)

        if chunks:
            print(f"💾 寫入 {len(chunks)} 個記憶節點到本地資料庫...")
            metadatas = [{"source": source_name} for _ in chunks]
            self.db.add_texts(texts=chunks, metadatas=metadatas)
            print("✅ 記憶存檔成功！")
            return True
        return False

    def crawl_and_ingest_url(self, url):
        try:
            print(f"🌐 正在前往抓取資訊: {url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            res.encoding = res.apparent_encoding
            soup = BeautifulSoup(res.text, 'html.parser')
            paragraphs = [p.text.strip() for p in soup.find_all('p') if len(p.text.strip()) > 20]
            full_text = "\n".join(paragraphs)

            if not full_text:
                print("⚠️ 找不到足夠的內文段落")
                return False

            return self.ingest_text_data(full_text, source_name=url)

        except Exception as e:
            print(f"❌ 網頁抓取失敗: {e}")
            return False

    def retrieve_memory(self, query, k=2):
        print(f"🧠 正在大腦深處搜尋關於「{query}」的相關資訊...")
        results = self.db.similarity_search(query, k=k)

        if not results:
            return "大腦中沒有找到相關記憶。"

        combined_context = "\n".join([
            f"情報來源({doc.metadata.get('source', '未知')}): {doc.page_content}"
            for doc in results
        ])
        return combined_context