import asyncio
import os
import httpx

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
        將文字轉為語音並儲存為 WAV 檔案。
        """
        print(f"\n🎙️ 正在呼叫 Groq TTS: 「{text}」")
        output_path = os.path.join(self.output_dir, filename)

        try:
            # 💡 你可以在 text 中加入情緒標籤測試，例如: "[cheerful] Hello everyone!"
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

                # 將回傳的音訊寫入檔案
                with open(output_path, "wb") as f:
                    f.write(response.content)

            file_size = os.path.getsize(output_path)
            print(f"✅ Groq 語音已儲存: {output_path} ({file_size} bytes)")
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
        print(f"✅ 生成成功: {path}")
    else:
        print("❌ 語音生成失敗")

if __name__ == "__main__":
    asyncio.run(main())