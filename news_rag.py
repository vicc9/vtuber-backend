import os
import requests
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class StreamerRAG:
    def __init__(self, db_path="./chroma_db"):
        print("🗄️ 初始化 ChromaDB 向量記憶庫...")
        self.db_path = db_path
        self.collection_name = "streamer_knowledge"
        self._embeddings = None  # ✅ 延遲載入，不在 __init__ 就載入模型
        self._db = None

    def _get_embeddings(self):
        if self._embeddings is None:
            print("📦 載入嵌入模型中...")
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        return self._embeddings

    def _get_db(self):
        if self._db is None:
            self._db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._get_embeddings(),
                persist_directory=self.db_path
            )
        return self._db

    def ingest_text_data(self, text_content, source_name="custom_input"):
        if not text_content.strip():
            print("⚠️ 沒有內容可以寫入！")
            return False

        print(f"✂️ 正在將文本進行記憶切片 ({source_name})...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_text(text_content)

        if chunks:
            print(f"💾 寫入 {len(chunks)} 個記憶節點...")
            metadatas = [{"source": source_name} for _ in chunks]
            self._get_db().add_texts(texts=chunks, metadatas=metadatas)
            print("✅ 記憶存檔成功！")
            return True
        return False

    def crawl_and_ingest_url(self, url):
        try:
            print(f"🌐 正在前往抓取資訊: {url}")
            headers = {"User-Agent": "Mozilla/5.0"}
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
        try:
            results = self._get_db().similarity_search(query, k=k)
            if not results:
                return "大腦中沒有找到相關記憶。"
            return "\n".join([
                f"情報來源({doc.metadata.get('source', '未知')}): {doc.page_content}"
                for doc in results
            ])
        except Exception as e:
            print(f"⚠️ 記憶檢索失敗: {e}")
            return "大腦中沒有找到相關記憶。"