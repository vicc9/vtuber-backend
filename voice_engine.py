import asyncio
import os
import httpx

class VoiceEngine:
    # Groq TTS 可用聲音列表（playai-tts 模型）
    # 中文友善推薦：Aaliyah、Nova、Chloe
    AVAILABLE_VOICES = [
        "Aaliyah", "Adelaide", "Angelo", "Arsenio", "Chloe",
        "Deedee", "Eleanor", "Fritz", "Gail", "George",
        "Indira", "Mamaw", "Mason", "Mikail", "Mitch",
        "Nova", "Quinn", "Ruby"
    ]

    def __init__(self, output_dir="./static/audio", voice="Nova"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.api_key = os.getenv("GROQ_API_KEY")
        # ✅ 使用不需要條款同意的 playai-tts 模型
        self.model = "playai-tts"
        self.voice = voice if voice in self.AVAILABLE_VOICES else "Nova"
        print(f"🎙️ VoiceEngine 初始化完成，模型: {self.model}，聲音: {self.voice}")

    async def text_to_speech(self, text: str, filename: str = "test_tts.wav") -> str | None:
        """
        將文字轉為語音並儲存為 WAV 檔案。
        回傳儲存路徑，失敗時回傳 None。
        """
        print(f"\n🎙️ 正在生成語音播報: 「{text}」")
        output_path = os.path.join(self.output_dir, filename)

        if not self.api_key:
            print("❌ 找不到 GROQ_API_KEY 環境變數")
            return None

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

                if response.status_code != 200:
                    print(f"❌ TTS API 錯誤 [{response.status_code}]: {response.text}")
                    return None

                with open(output_path, "wb") as f:
                    f.write(response.content)

            file_size = os.path.getsize(output_path)
            print(f"✅ 語音已儲存: {output_path} ({file_size} bytes)")
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