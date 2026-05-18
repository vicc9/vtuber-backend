import os
import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from dotenv import load_dotenv

from news_rag import StreamerRAG
from voice_engine import VoiceEngine
from streamer_brain import StreamerBrain
from stt_engine import transcribe_audio
from idle_manager import get_idle_prompt
from behavior_tracker import BehaviorTracker, load_behavior_context
from auth_manager import generate_token, verify_token

load_dotenv()

WEBGL_DIR = "D:/VTuber_WebGL_Build"

# ─────────────────────────────────────────
# 系統初始化
# ─────────────────────────────────────────
print("初始化系統中，請稍候...")
rag_system   = StreamerRAG()
voice_engine = VoiceEngine()
brain        = StreamerBrain()
tracker      = BehaviorTracker()
behavior_ctx = {}

# ─────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global behavior_ctx
    behavior_ctx = await load_behavior_context()
    print(f"📊 行為記憶載入完成：{behavior_ctx}")
    yield
    print("\n🔄 伺服器關閉中...")
    await tracker.save_session()
    summary = await brain.generate_session_summary()
    if summary:
        print(f"📝 本場摘要：{summary}")
    else:
        print("📭 本場無摘要可儲存")

# ─────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

os.makedirs("./static/audio", exist_ok=True)
app.mount("/audio", StaticFiles(directory="./static/audio"), name="audio")

# ─────────────────────────────────────────
# ① API 路由（必須在靜態路由之前定義）
# ─────────────────────────────────────────
@app.get("/api/token")
async def get_token(client_id: str = Query(default="default")):
    """前端啟動時呼叫一次，取得帶時效的 HMAC Token"""
    token = generate_token(client_id)
    print(f"[Auth] 發放 Token 給 client_id={client_id}")
    return JSONResponse({"token": token})

# ─────────────────────────────────────────
# ② WebSocket 端點
# ─────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(default="")
):
    # 驗證 Token
    if not verify_token(token, client_id="vtuber_app"):
        print(f"[Auth] ❌ Token 無效，拒絕連線")
        await websocket.close(code=4001)
        return

    await websocket.accept()
    print("🌐 前端已連線（已驗證）！")

    thresholds = {
        "first":  behavior_ctx.get("first_threshold",  15),
        "bored":  behavior_ctx.get("bored_threshold",  45),
        "sleepy": behavior_ctx.get("sleepy_threshold", 120),
    }
    stages           = ["first", "bored", "sleepy"]
    idle_stage_index = 0
    idle_task        = None
    silence_start    = asyncio.get_event_loop().time()

    def schedule_idle():
        nonlocal idle_task, idle_stage_index
        if idle_stage_index >= len(stages):
            return
        stage = stages[idle_stage_index]
        wait  = thresholds[stage]

        async def idle_trigger():
            try:
                await asyncio.sleep(wait)
                nonlocal idle_stage_index
                await send_idle_response(websocket, stage)
                idle_stage_index += 1
                schedule_idle()
            except asyncio.CancelledError:
                pass

        idle_task = asyncio.create_task(idle_trigger())

    def reset_idle_timer(input_type: str = "text"):
        nonlocal idle_task, idle_stage_index, silence_start
        elapsed = asyncio.get_event_loop().time() - silence_start
        tracker.record_silence(elapsed)
        tracker.record_input(input_type)
        idle_stage_index = 0
        silence_start    = asyncio.get_event_loop().time()
        if idle_task and not idle_task.done():
            idle_task.cancel()
        schedule_idle()

    schedule_idle()

    try:
        while True:
            message = await websocket.receive()

            if message.get("bytes"):
                audio_bytes = message["bytes"]
                print(f"🎤 收到音訊，大小: {len(audio_bytes)} bytes")
                reset_idle_timer("voice")

                await websocket.send_text(json.dumps({
                    "type": "stt_status", "status": "recognizing"
                }))

                user_text = await transcribe_audio(audio_bytes)

                if not user_text:
                    await websocket.send_text(json.dumps({
                        "type":    "stt_status",
                        "status":  "failed",
                        "message": "無法辨識語音，請再試一次"
                    }))
                    reset_idle_timer("voice")
                    continue

                await websocket.send_text(json.dumps({
                    "type": "stt_result", "text": user_text
                }, ensure_ascii=False))

                print(f"👤 語音輸入辨識結果: {user_text}")
                await process_and_respond(websocket, user_text)
                reset_idle_timer("voice")

            elif message.get("text"):
                user_msg = message["text"].strip()
                if not user_msg:
                    continue
                print(f"\n👤 觀眾說: {user_msg}")
                reset_idle_timer("text")
                await process_and_respond(websocket, user_msg)
                reset_idle_timer("text")

    except WebSocketDisconnect:
        print("🔌 前端已斷線。")
    except Exception as e:
        print(f"❌ WebSocket 錯誤: {e}")
    finally:
        if idle_task and not idle_task.done():
            idle_task.cancel()

# ─────────────────────────────────────────
# ③ 靜態路由（必須最後定義，避免攔截 API）
# ─────────────────────────────────────────
@app.get("/")
async def serve_root():
    return FileResponse(os.path.join(WEBGL_DIR, "index.html"))

@app.get("/{full_path:path}")
async def serve_static(full_path: str):
    file_path = os.path.join(WEBGL_DIR, full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join(WEBGL_DIR, "index.html"))

# ─────────────────────────────────────────
# 共用：LLM + TTS + 推播
# ─────────────────────────────────────────
async def process_and_respond(websocket: WebSocket, user_msg: str):
    context = rag_system.retrieve_memory(query=user_msg, k=2)
    print(f"🧠 正在搜尋關於「{user_msg}」的相關記憶...")

    behavior_prompt  = behavior_ctx.get("behavior_prompt", "")
    combined_context = context
    if behavior_prompt:
        combined_context = f"{context}\n\n## 使用者習慣\n{behavior_prompt}"

    response_data = await brain.generate_live_response(
        user_input=user_msg,
        rag_context=combined_context,
        speaker_name="觀眾"
    )
    print(f"🧠 大腦決策: {json.dumps(response_data, ensure_ascii=False)}")

    audio_filename = f"live_{int(time.time())}.wav"
    await voice_engine.text_to_speech(
        text=response_data["dialogue"],
        filename=audio_filename
    )

    payload = {
        "type":      "streamer_update",
        "dialogue":  response_data["dialogue"],
        "emotion":   response_data["emotion"],
        "action":    response_data["action"],
        "audio_url": f"http://localhost:8000/audio/{audio_filename}",
        "is_idle":   False,
    }
    await websocket.send_text(json.dumps(payload, ensure_ascii=False))
    print("📦 回應已發送給前端！")

# ─────────────────────────────────────────
# 共用：Idle 自動搭話
# ─────────────────────────────────────────
async def send_idle_response(websocket: WebSocket, stage: str):
    intimacy = behavior_ctx.get("intimacy_score", 0)
    idle     = get_idle_prompt(stage, intimacy)
    print(f"[Idle] stage={stage}，觸發：{idle['text']}")

    audio_filename = f"idle_{int(time.time())}.wav"
    await voice_engine.text_to_speech(
        text=idle["text"],
        filename=audio_filename
    )

    payload = {
        "type":      "streamer_update",
        "dialogue":  idle["text"],
        "emotion":   idle["emotion"],
        "action":    idle["action"],
        "audio_url": f"http://localhost:8000/audio/{audio_filename}",
        "is_idle":   True,
        "stage":     stage,
    }
    await websocket.send_text(json.dumps(payload, ensure_ascii=False))

# ─────────────────────────────────────────
# 啟動入口
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 啟動 AI 主播伺服器中...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
