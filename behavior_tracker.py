# D:\VTuber_Backend\behavior_tracker.py
import os
from datetime import date
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


class BehaviorTracker:
    """
    追蹤本次場次的使用者行為。
    結束時寫入 Supabase，下次啟動時讀取作為個性化依據。
    """

    def __init__(self):
        self.silence_records: list[float] = []
        self.total_inputs:  int   = 0
        self.voice_inputs:  int   = 0
        self.text_inputs:   int   = 0

    def record_silence(self, seconds: float):
        if seconds > 1.0:   # 過濾系統初始化的雜訊
            self.silence_records.append(seconds)

    def record_input(self, input_type: str):
        """input_type: 'voice' or 'text'"""
        self.total_inputs += 1
        if input_type == "voice":
            self.voice_inputs += 1
        else:
            self.text_inputs += 1

    def get_user_type(self) -> str:
        if not self.silence_records:
            return "unknown"
        avg = sum(self.silence_records) / len(self.silence_records)
        if avg < 8:
            return "active"     # 活潑型：快速回應
        elif avg < 20:
            return "normal"     # 正常型
        elif avg < 60:
            return "silent"     # 沉默型：常常不搭話
        else:
            return "busy"       # 忙碌型：很少回應

    async def save_session(self):
        """場次結束時寫入 Supabase"""
        if not self.silence_records:
            return

        avg_s       = sum(self.silence_records) / len(self.silence_records)
        max_s       = max(self.silence_records)
        voice_ratio = self.voice_inputs / self.total_inputs if self.total_inputs > 0 else 0
        user_type   = self.get_user_type()

        # 讀取舊親密度，累積增加
        intimacy_delta = min(self.total_inputs * 2, 20)
        old = supabase.table("user_behavior") \
            .select("intimacy_score") \
            .order("created_at", desc=True) \
            .limit(1).execute()
        old_score = old.data[0]["intimacy_score"] if old.data else 0
        new_score = min(old_score + intimacy_delta, 100)

        supabase.table("user_behavior").insert({
            "session_date":    str(date.today()),
            "avg_silence_sec": round(avg_s, 1),
            "max_silence_sec": round(max_s, 1),
            "total_inputs":    self.total_inputs,
            "voice_ratio":     round(voice_ratio, 2),
            "user_type":       user_type,
            "intimacy_score":  new_score,
        }).execute()

        print(f"[Behavior] 儲存完成：類型={user_type}，"
              f"平均沉默={avg_s:.1f}s，親密度={new_score}")


async def load_behavior_context() -> dict:
    """
    啟動時讀取最近 5 次場次，
    回傳動態閾值與個性化 prompt。
    """
    try:
        result = supabase.table("user_behavior") \
            .select("*") \
            .order("created_at", desc=True) \
            .limit(5).execute()

        if not result.data:
            return _default_context()

        records  = result.data
        latest   = records[0]
        avg_all  = sum(r["avg_silence_sec"] for r in records) / len(records)
        user_type = latest["user_type"]
        intimacy  = latest["intimacy_score"]

        # 動態調整沉默閾值
        if avg_all < 8:
            first_threshold = 20    # 活潑型：等久一點
        elif avg_all < 15:
            first_threshold = 15
        else:
            first_threshold = max(int(avg_all * 0.6), 5)  # 沉默型：提早搭話

        behavior_prompt = _build_behavior_prompt(user_type, intimacy, avg_all, latest)

        return {
            "first_threshold":  first_threshold,
            "bored_threshold":  first_threshold * 3,
            "sleepy_threshold": first_threshold * 8,
            "user_type":        user_type,
            "intimacy_score":   intimacy,
            "behavior_prompt":  behavior_prompt,
            "avg_silence":      avg_all,
        }

    except Exception as e:
        print(f"[Behavior] 讀取失敗: {e}")
        return _default_context()


def _build_behavior_prompt(user_type, intimacy, avg_silence, latest) -> str:
    lines = []

    type_desc = {
        "active": "這位使用者很活潑，習慣快速回應，你可以語氣輕鬆愉快。",
        "silent": f"這位使用者習慣沉默（平均 {avg_silence:.0f} 秒才回應），"
                  "你要更主動搭話，但不要讓對方有壓力。",
        "busy":   "這位使用者可能很忙，常常沒有回應，你的搭話要簡短有趣。",
        "normal": "這位使用者互動節奏正常。",
    }.get(user_type, "")
    if type_desc:
        lines.append(type_desc)

    if intimacy >= 70:
        lines.append("你們已經很熟了，可以用更親密的語氣，偶爾嘲諷或開玩笑都沒關係。")
    elif intimacy >= 40:
        lines.append("你們有一定的熟悉度，可以適當開玩笑，但還是保持禮貌。")
    else:
        lines.append("你們還不太熟，態度要友善有禮，先建立信任感。")

    if avg_silence > 30:
        lines.append(
            f"特別注意：這位使用者上次平均沉默 {avg_silence:.0f} 秒。"
            "如果對方又沉默了，你可以用帶點嘲諷但關心的語氣問他是不是忙或是睡著了。"
        )

    if latest["max_silence_sec"] > 60:
        lines.append(
            f"上次對方最長沉默了 {latest['max_silence_sec']:.0f} 秒才回應，"
            "你可以在適當時機提起這件事，例如『你上次也消失很久耶』。"
        )

    return "\n".join(lines)


def _default_context() -> dict:
    return {
        "first_threshold":  15,
        "bored_threshold":  45,
        "sleepy_threshold": 120,
        "user_type":        "unknown",
        "intimacy_score":   0,
        "behavior_prompt":  "這是你們第一次見面，態度友善有禮，慢慢認識對方。",
        "avg_silence":      0,
    }