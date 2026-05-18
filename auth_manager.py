# D:\VTuber_Backend\auth_manager.py
import os
import hmac
import hashlib
import time
from dotenv import load_dotenv

load_dotenv()

APP_SECRET = os.getenv("APP_SECRET", "change_this_to_random_string")


def generate_token(client_id: str = "default") -> str:
    """
    產生帶時效的 HMAC Token。
    格式：timestamp:hmac_signature
    有效期：24 小時
    """
    timestamp = str(int(time.time()))
    message   = f"{client_id}:{timestamp}"
    signature = hmac.new(
        APP_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    token = f"{timestamp}:{signature}"
    print(f"[Auth] 產生 Token，timestamp={timestamp}")
    return token


def verify_token(token: str, client_id: str = "default",
                 max_age_seconds: int = 86400) -> bool:
    """
    驗證 Token 是否合法且未過期。
    """
    if not token:
        print("[Auth] Token 為空")
        return False

    try:
        parts = token.split(":")
        if len(parts) != 2:
            print(f"[Auth] Token 格式錯誤，parts={len(parts)}")
            return False

        timestamp_str, received_sig = parts
        timestamp = int(timestamp_str)

        # 檢查是否過期
        age = time.time() - timestamp
        if age > max_age_seconds:
            print(f"[Auth] Token 已過期，age={age:.0f}s")
            return False

        # 重新計算 HMAC（與 generate_token 完全相同的邏輯）
        message      = f"{client_id}:{timestamp_str}"
        expected_sig = hmac.new(
            APP_SECRET.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        # 安全比對（防止 timing attack）
        result = hmac.compare_digest(received_sig, expected_sig)
        if not result:
            print(f"[Auth] HMAC 比對失敗")
            print(f"  received : {received_sig[:16]}...")
            print(f"  expected : {expected_sig[:16]}...")
        return result

    except Exception as e:
        print(f"[Auth] Token 驗證例外: {e}")
        return False


if __name__ == "__main__":
    # 本地測試
    print("=== 測試 auth_manager ===")
    token = generate_token("vtuber_app")
    print(f"產生 Token: {token}")
    result = verify_token(token, "vtuber_app")
    print(f"驗證結果: {result}")
    result2 = verify_token("fake:token", "vtuber_app")
    print(f"假 Token 驗證: {result2}")