import asyncio
import os
import io
import wave
from groq import AsyncGroq
import httpx

class VoiceEngine:
    def __init__(self, output_dir="./static/audio"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.api_key = os.getenv("GROQ_API_KEY")

    async def text_to_speech(self, text, filename="test_tts.wav"):
        print(f"\n🎙️ 正在生成語音播報: 「{text}」")
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
                        # ✅ 已替換為 Groq 目前支援的 Orpheus 模型與聲音
                        "model": "canopylabs/orpheus-v1-english", 
                        "voice": "autumn", 
                        "input": text,
                    }
                )

                # 印出詳細錯誤方便 debug
                if response.status_code != 200:
                    print(f"❌ TTS 錯誤回應: {response.status_code} - {response.text}")
                    response.raise_for_status()

                with open(output_path, "wb") as f:
                    f.write(response.content)

            print(f"✅ 語音檔案已成功儲存至: {output_path}")

        except Exception as e:
            print(f"❌ Groq TTS 失敗: {e}")
            # 這裡回傳空字串或 None，避免整個後端因為語音失敗而崩潰導致斷線
            return None 

        return output_path


async def main():
    engine = VoiceEngine()
    tts_audio_path = await engine.text_to_speech("大家好，我是今天的值班主播小光！")
    if tts_audio_path:
        print(f"生成的音訊位置: {tts_audio_path}")
    else:
        print("語音生成失敗。")

if __name__ == "__main__":
    asyncio.run(main())