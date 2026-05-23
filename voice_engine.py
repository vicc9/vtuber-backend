import asyncio
import os
import io
import wave
from groq import AsyncGroq

class VoiceEngine:
    def __init__(self, output_dir="./static/audio"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.tts_voice = "Celeste-PlayAI"  # 中文效果較好的聲音

    async def text_to_speech(self, text, filename="test_tts.wav"):
        print(f"\n🎙️ 正在生成語音播報: 「{text}」")
        output_path = os.path.join(self.output_dir, filename)

        try:
            response = await self.client.audio.speech.create(
                model="playai-tts",
                voice=self.tts_voice,
                input=text,
                response_format="wav",
            )
            # ✅ 改用 with_streaming_response 避免 aread() 相容性問題
            with open(output_path, "wb") as f:
                f.write(response.content)

            print(f"✅ 語音檔案已成功儲存至: {output_path}")

        except Exception as e:
            print(f"❌ Groq TTS 失敗: {e}")
            raise

        return output_path


async def main():
    engine = VoiceEngine()
    tts_audio_path = await engine.text_to_speech("大家好，我是今天的值班主播小光！")
    print(f"生成的音訊位置: {tts_audio_path}")

if __name__ == "__main__":
    asyncio.run(main())