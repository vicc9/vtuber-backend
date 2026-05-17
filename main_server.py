import os
import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import uvicorn
from dotenv import load_dotenv

from news_rag import StreamerRAG
from voice_engine import VoiceEngine
from streamer_brain import StreamerBrain

load_dotenv()

# ─────────────────────────────────────────
# 系統初始化（在 lifespan 外先建立，確保跨請求共用同一份實例）
# ─────────────────────────────────────────
print("初始化系統中，請稍候...")
rag_system = StreamerRAG()
voice_engine = VoiceEngine()
brain = StreamerBrain()

# ─────────────────────────────────────────
# Lifespan：伺服器啟動 / 關閉的生命週期
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 伺服器啟動後執行的事情（目前不需要額外動作，初始化已在上方完成）
    yield
    # ── 伺服器關閉時：自動生成並儲存場次摘要 ──
    print("\n🔄 伺服器關閉中，正在生成場次摘要...")
    summary = await brain.generate_session_summary()
    if summary:
        print(f"📝 本場摘要：{summary}")
    else:
        print("📭 本場無摘要可儲存")

# ─────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

# 建立並掛載靜態音訊資料夾，讓 Unity 可以透過網址下載播報語音
os.makedirs("./static/audio", exist_ok=True)
app.mount("/audio", StaticFiles(directory="./static/audio"), name="audio")


# ─────────────────────────────────────────
# WebSocket 端點
# ─────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🌐 Unity 前端已連線！全雙工通道開啟。")

    try:
        while True:
            # 1. 接收來自 Unity 的訊息
            user_msg = await websocket.receive_text()
            print(f"\n👤 觀眾說: {user_msg}")

            # 2. 觸發 RAG，去記憶庫檢索相關資訊
            context = rag_system.retrieve_memory(query=user_msg, k=2)
            print(f"🧠 正在大腦深處搜尋關於「{user_msg}」的相關資訊...")

            # 3. 大腦思考，產出文稿、表情與動作的 JSON 決策
            #    speaker_name 目前預設「觀眾」，步驟 11 接直播平台後可傳真實名稱
            response_data = await brain.generate_live_response(
                user_input=user_msg,
                rag_context=context,
                speaker_name="觀眾"
            )
            print(f"🧠 主播大腦決策: {json.dumps(response_data, ensure_ascii=False)}")

            # 4. 合成 TTS 語音檔
            timestamp = int(time.time())
            audio_filename = f"live_{timestamp}.wav"
            print(f"🎙️ 正在生成語音播報: 「{response_data['dialogue']}」")
            await voice_engine.text_to_speech(
                text=response_data["dialogue"],
                filename=audio_filename
            )
            print(f"✅ 語音檔案已成功儲存至: ./static/audio/{audio_filename}")

            # 5. 組合完整音訊網址
            full_audio_url = f"http://localhost:8000/audio/{audio_filename}"

            # 6. 打包成 Unity 看得懂的標準包裹，推播回前端
            payload = {
                "type": "streamer_update",
                "dialogue": response_data["dialogue"],
                "emotion": response_data["emotion"],
                "action": response_data["action"],
                "audio_url": full_audio_url
            }
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
            print("📦 回應已成功發送給 Unity！")

    except WebSocketDisconnect:
        print("🔌 Unity 已經斷線。")
    except Exception as e:
        print(f"❌ WebSocket 發生錯誤: {e}")


# ─────────────────────────────────────────
# 啟動入口
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 啟動 AI 主播伺服器中...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
