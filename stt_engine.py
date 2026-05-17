# D:\VTuber_Backend\stt_engine.py
import os
import tempfile
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))


async def transcribe_audio(audio_bytes: bytes, language: str = "zh") -> str:
    """
    接收 WAV bytes，回傳辨識文字。
    使用 Groq whisper-large-v3（速度快、支援中文）
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            transcription = await client.audio.transcriptions.create(
                file=("audio.wav", f, "audio/wav"),
                model="whisper-large-v3",
                language=language,
                response_format="text",
                temperature=0.0,
            )

        os.unlink(tmp_path)
        result = transcription.strip()
        print(f"[STT] 辨識結果: {result}")
        return result

    except Exception as e:
        print(f"[STT] 辨識失敗: {e}")
        return ""