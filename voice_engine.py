import asyncio
import os
import httpx
import soundfile as sf
import io

class VoiceEngine:
    # 🌟 Groq 最新 Orpheus TTS 可用英文聲音列表
    # 女性: autumn, diana, hannah
    # 男性: austin, daniel, troy
    AVAILABLE_VOICES = ["autumn", "diana", "hannah", "austin", "daniel", "troy"]

    def __init__(self, output_dir="./static/audio", voice="diana"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.api_key = os.getenv("GROQ_API_KEY")
        
        # ✅ 改用 Groq 官方最新推出的 Canopy Labs Orpheus 模型
        self.model = "canopylabs/orpheus-v1-english"
        self.voice = voice if voice in self.AVAILABLE_VOICES else "diana"
        print(f"🎙️ VoiceEngine 初始化，使用 Groq 最新模型: {self.model}，聲音: {self.voice}")

    async def text_to_speech(self, text: str, filename: str = "test_tts.wav") -> str | None:
        """
        將文字轉為語音並儲存為標準且兼容 Unity 的 WAV 檔案。
        """
        print(f"\n🎙️ 正在呼叫 Groq TTS: 「{text}」")
        output_path = os.path.join(self.output_dir, filename)

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/audio/speech",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "voice": self.voice,
                        "input": text,
                        "response_format": "wav",
                    }
                )

                # 捕捉並印出詳細的 API 錯誤，方便除錯
                if response.status_code != 200:
                    print(f"❌ Groq TTS API 錯誤 [{response.status_code}]: {response.text}")
                    return None

                # ✅ 關鍵修復：使用 soundfile 將串流 WAV 重新編碼為 Unity 100% 支援的標準 16-bit PCM WAV
                # 這樣可以修復 API 傳回的 WAV 檔頭長度為 0 的問題
                audio_data, sample_rate = sf.read(io.BytesIO(response.content))
                sf.write(output_path, audio_data, sample_rate, subtype='PCM_16')

            file_size = os.path.getsize(output_path)
            print(f"✅ Groq 語音已重新編碼並儲存: {output_path} ({file_size} bytes)")
            return output_path

        except httpx.TimeoutException:
            print("❌ Groq TTS 請求超時（30秒）")
            return None
        except Exception as e:
            print(f"❌ Groq TTS 失敗: {e}")
            return None

async def main():
    engine = VoiceEngine()
    path = await engine.text_to_speech("大家好，我是今天的值班主播！")
    if path:
        print("測試成功！")

if __name__ == "__main__":
    asyncio.run(main())