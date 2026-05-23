import asyncio
import os
import edge_tts
import soundfile as sf
import io

class VoiceEngine:
    def __init__(self, output_dir="./static/audio"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 移除了 Faster-Whisper 的初始化，讓伺服器啟動更快、更省記憶體
        self.tts_voice = "zh-CN-XiaoyiNeural"

    async def text_to_speech(self, text, filename="test_tts.wav"):
        print(f"\n🎙️ 正在生成語音播報: 「{text}」")
        output_path = os.path.join(self.output_dir, filename)
        
        # 用串流方式收集 edge-tts 的音訊 bytes
        communicate = edge_tts.Communicate(text, self.tts_voice, rate="+5%")
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        
        # 嘗試用 soundfile 轉成 WAV
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            data, samplerate = sf.read(audio_buffer)
            sf.write(output_path, data, samplerate)
            print(f"✅ 語音檔案已成功儲存至: {output_path}")
        except Exception as e:
            print(f"❌ soundfile 轉換失敗: {e}")
            raise
        
        return output_path

async def main():
    engine = VoiceEngine()
    tts_audio_path = await engine.text_to_speech("大家好，我是今天的值班主播小光！")
    print(f"生成的音訊位置: {tts_audio_path}")

if __name__ == "__main__":
    asyncio.run(main())