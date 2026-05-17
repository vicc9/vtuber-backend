# D:\VTuber_Backend\idle_manager.py
import random

# 依親密度分三層的搭話台詞
IDLE_PROMPTS = {
    "first": {
        "low": [        # 親密度 < 40，還不熟
            "嗯⋯好安靜，有什麼想聊的嗎？",
            "你在嗎？我在這裡喔！",
            "有沒有什麼想問我的事呢？",
        ],
        "mid": [        # 親密度 40~70
            "欸，你消失去哪了？",
            "在嗎在嗎？別讓我一個人說話啦！",
            "你是在偷偷看我嗎（笑）",
        ],
        "high": [       # 親密度 > 70，很熟了
            "喂，你又發呆了喔。",
            "你是不是又在想奇怪的事（笑）",
            "我都等到無聊了耶，快說點什麼！",
        ],
    },
    "bored": {
        "low": [
            "（輕嘆氣）沒關係，你慢慢來⋯我等你。",
            "好安靜⋯你還在嗎？",
        ],
        "mid": [
            "（伸懶腰）你到底去哪了啦⋯",
            "等等等，你是出去買飲料沒有跟我說嗎？",
        ],
        "high": [
            "（嘆氣）好啦我知道你在忙，但你能不能說一聲啊！",
            "你上次也這樣，消失一大段時間（翻白眼）",
        ],
    },
    "sleepy": {
        "low":  ["（打哈欠）好睏⋯你還在嗎⋯"],
        "mid":  ["（快睡著）zz⋯啊！我沒睡！你呢？"],
        "high": ["你再不說話我真的睡了喔，不是開玩笑的（眯眼）"],
    },
}

# emotion / action 對應（與 MotionController 一致）
IDLE_STATES = {
    "first":  {"emotion": "curious", "action": "Wave"},
    "bored":  {"emotion": "sad",     "action": "Think"},
    "sleepy": {"emotion": "sleepy",  "action": "Nod"},
}


def get_idle_prompt(stage: str, intimacy_score: int = 0) -> dict:
    """
    回傳指定階段的自動搭話內容。
    intimacy_score: 0~100，決定台詞親密度。
    """
    if intimacy_score >= 70:
        tier = "high"
    elif intimacy_score >= 40:
        tier = "mid"
    else:
        tier = "low"

    pool  = IDLE_PROMPTS.get(stage, IDLE_PROMPTS["first"])
    lines = pool.get(tier, pool["low"])
    state = IDLE_STATES.get(stage, IDLE_STATES["first"])

    return {
        "text":    random.choice(lines),
        "emotion": state["emotion"],
        "action":  state["action"],
        "stage":   stage,
    }