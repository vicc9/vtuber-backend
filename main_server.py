import tempfile, os
import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware # 🌟 部署修改：匯入 CORS
import uvicorn
from dotenv import load_dotenv

# ✅ 第三步修改：改為從我們新建的 supabase_rag 引入
from supabase_rag import StreamerRAG 
from voice_engine import VoiceEngine
from streamer_brain import StreamerBrain
from stt_engine import transcribe_audio
from idle_manager import get_idle_prompt
from behavior_tracker import BehaviorTracker, load_behavior_context
from auth_manager import generate_token, verify_token
from fastapi import UploadFile, File

load_dotenv()

# ─────────────────────────────────────────
# 系統初始化
# ─────────────────────────────────────────
print("初始化系統中，請稍候...")

# 🌟 修正 404 路徑問題：取得當前檔案的絕對路徑，確保音訊資料夾位置正確
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# ✅ 初始化輕量版的 Supabase RAG
rag_system   = StreamerRAG()
# ✅ 強制傳入絕對路徑給 VoiceEngine，確保儲存位置正確
voice_engine = VoiceEngine(output_dir=AUDIO_DIR)
brain        = StreamerBrain()
tracker      = BehaviorTracker()
behavior_ctx = {}

# ─────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global behavior_ctx

    # ✅ Startup：用 try/except 保護，Supabase 失敗不影響服務啟動
    try:
        behavior_ctx = await load_behavior_context()
        print(f"📊 行為記憶載入完成：{behavior_ctx}")
    except Exception as e:
        print(f"⚠️ 行為記憶載入失敗（使用預設值）: {e}")
        behavior_ctx = {
            "first_threshold":  15,
            "bored_threshold":  45,
            "sleepy_threshold": 120,
            "user_type":        "unknown",
            "intimacy_score":   0,
            "behavior_prompt":  "這是你們第一次見面，態度友善有禮，慢慢認識對方。",
            "avg_silence":      0,
        }

    yield

    # ✅ Shutdown：用 try/except 保護，Supabase 失敗不阻止正常退出
    print("\n🔄 伺服器關閉中...")
    try:
        await tracker.save_session()
    except Exception as e:
        print(f"⚠️ 行為記憶儲存失敗: {e}")

    try:
        summary = await brain.generate_session_summary()
        if summary:
            print(f"📝 本場摘要：{summary}")
        else:
            print("📭 本場無摘要可儲存")
    except Exception as e:
        print(f"⚠️ 摘要生成失敗: {e}")

# ─────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

# 🌟 部署修改：加入 CORS 設定，允許前端跨網域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🌟 修正 404 路徑問題：改用絕對路徑掛載靜態資料夾
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# ─────────────────────────────────────────
# ① API 路由
# ─────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "VTuber Backend is running!"}

@app.get("/api/token")
async def get_token():
    client_id = "vtuber_app"   
    token = generate_token(client_id)
    print(f"[Auth] 發放 Token 給 client_id={client_id}")
    return JSONResponse({"token": token})

@app.post("/api/stt")
async def stt_endpoint(audio: UploadFile = File(...)):
    suffix = ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        text = await transcribe_audio(tmp_path) 
        return {"text": text}
    finally:
        os.unlink(tmp_path)

# ─────────────────────────────────────────
# ② WebSocket 端點
# ─────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(default="")
):
    if not verify_token(token, client_id="vtuber_app"):
        print(f"[Auth] ❌ Token 無效，拒絕連線")
        await websocket.close(code=4001)
        return

    await websocket.accept()
    print(f"[WS] ✅ 新連線已接受")

    send_lock = asyncio.Lock()

    async def safe_send_json(data: dict):
        async with send_lock:
            try:
                await websocket.send_text(json.dumps(data, ensure_ascii=False))
            except Exception as e:
                print(f"⚠️ 鎖定發送時發生異常 (可能連線已關閉): {e}")

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
                await send_idle_response(websocket, stage, send_lock)
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

            if message.get("type") == "websocket.disconnect":
                print(f"🔌 前端已正常/異常中斷連線 (Code: {message.get('code', 1000)})")
                break

            if message.get("bytes"):
                audio_bytes = message["bytes"]
                print(f"🎤 收到音訊，大小: {len(audio_bytes)} bytes")
                reset_idle_timer("voice")

                await safe_send_json({"type": "stt_status", "status": "recognizing"})

                user_text = await transcribe_audio(audio_bytes)

                if not user_text:
                    await safe_send_json({
                        "type":    "stt_status",
                        "status":  "failed",
                        "message": "無法辨識語音，請再試一次"
                    })
                    reset_idle_timer("voice")
                    continue

                await safe_send_json({"type": "stt_result", "text": user_text})

                print(f"👤 語音輸入辨識結果: {user_text}")
                await process_and_respond(websocket, user_text, send_lock)
                reset_idle_timer("voice")

            elif message.get("text"):
                user_msg = message["text"].strip()
                if not user_msg:
                    continue
                print(f"\n👤 觀眾說: {user_msg}")
                reset_idle_timer("text")
                await process_and_respond(websocket, user_msg, send_lock)
                reset_idle_timer("text")

    except WebSocketDisconnect:
        print("🔌 前端已斷線。")
    except Exception as e:
        print(f"❌ WebSocket 錯誤: {e}")
    finally:
        if idle_task and not idle_task.done():
            idle_task.cancel()
        print("🏁 WebSocket 清理程序與資源釋放完成。")


# ─────────────────────────────────────────
# 共用：LLM + TTS + 推播
# ─────────────────────────────────────────
async def process_and_respond(websocket: WebSocket, user_msg: str, send_lock: asyncio.Lock):
    
    # ✅ 第三步修改：加上 await，因為我們新寫的 Supabase RAG 是非同步(async)的
    print(f"🧠 正在 Supabase 搜尋關於「{user_msg}」的相關記憶...")
    context = await rag_system.retrieve_memory(query=user_msg, k=2)

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
    saved_path = await voice_engine.text_to_speech(
        text=response_data["dialogue"],
        filename=audio_filename
    )

    payload = {
        "type":      "streamer_update",
        "dialogue":  response_data["dialogue"],
        "emotion":   response_data["emotion"],
        "action":    response_data["action"],
        "audio_url": f"https://vtuber-backend-qwmt.onrender.com/audio/{audio_filename}" if saved_path else "",
        "is_idle":   False,
    }

    async with send_lock:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))
    print("📦 回應已發送（含音訊）！" if saved_path else "📦 回應已發送（無音訊，TTS 失敗）！")

# ─────────────────────────────────────────
# 共用：Idle 自動搭話
# ─────────────────────────────────────────
async def send_idle_response(websocket: WebSocket, stage: str, send_lock: asyncio.Lock):
    intimacy = behavior_ctx.get("intimacy_score", 0)
    idle     = get_idle_prompt(stage, intimacy)
    print(f"[Idle] stage={stage}，觸發：{idle['text']}")

    audio_filename = f"idle_{int(time.time())}.wav"
    saved_path = await voice_engine.text_to_speech(
        text=idle["text"],
        filename=audio_filename
    )

    payload = {
        "type":      "streamer_update",
        "dialogue":  idle["text"],
        "emotion":   idle["emotion"],
        "action":    idle["action"],
        "audio_url": f"https://vtuber-backend-qwmt.onrender.com/audio/{audio_filename}" if saved_path else "",
        "is_idle":   True,
        "stage":     stage,
    }

    async with send_lock:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))

# ─────────────────────────────────────────
# 啟動入口
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 啟動 AI 主播伺服器中...")
    uvicorn.run(app, host="0.0.0.0", port=8000)