"""OpenAI 兼容 API 客户端（多平台 failover + 重试）。

平台配置来自 config.DEEPSEEK_PLATFORMS：每项包含 key / base_url / model / quota。
无 _N 后缀配置时退回 DEEPSEEK_API_KEY + API_BASE_URL 单平台变量。
"""
import os
import time
import threading

import requests
from dotenv import load_dotenv

_ = load_dotenv()

# 可重试状态码：限流/网关类瞬时故障，重试通常能自愈。
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _platform_headers(platform: dict) -> dict:
    return {"Authorization": f"Bearer {platform['key']}"}


class ApiClient:
    _instance = None
    _instance_lock = threading.Lock()
    # 类级平台黑名单：401/403（key 无效/余额不足）永久跳过，避免每次先撞失败平台
    _dead_platforms = set()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
        return cls._instance

    def _init(self):
        from config import API_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_PLATFORMS

        if DEEPSEEK_PLATFORMS:
            self.platforms = [
                {"key": p.key, "base_url": p.base_url, "model": p.model,
                 "quota": p.quota, "name": p.name}
                for p in DEEPSEEK_PLATFORMS
            ]
        else:
            self.platforms = [{
                "key": DEEPSEEK_API_KEY,
                "base_url": API_BASE_URL,
                "model": os.getenv("API_MODEL", "deepseek-chat"),
                "quota": 0,
                "name": "legacy",
            }]

        # 每平台独立并发闸（quota=0 → 不限制，仅受请求超时约束）
        self._gates = {
            i: threading.BoundedSemaphore(p["quota"]) if p["quota"] > 0 else None
            for i, p in enumerate(self.platforms)
        }

        # 代理侧超时/重试参数全部走环境变量：
        #   API_TIMEOUT          单请求超时秒数（默认 900）
        #   API_MAX_RETRIES      重试次数（默认 2，总请求数 = 次数 + 1）
        #   API_RETRY_BACKOFF    重试基础退避秒数（默认 2，指数退避）
        #   API_RETRY_STATUSES   逗号分隔的可重试 HTTP 状态码
        self.timeout = int(os.getenv("API_TIMEOUT", "300"))
        self.max_retries = int(os.getenv("API_MAX_RETRIES", "2"))
        self.retry_backoff = float(os.getenv("API_RETRY_BACKOFF", "2.0"))
        self.retry_statuses = {
            int(x) for x in os.getenv(
                "API_RETRY_STATUSES", "408,429,500,502,503,504"
            ).split(",") if x.strip().isdigit()
        } or _RETRYABLE_STATUS

        self.session = requests.Session()
        # Authorization 不能放 session 级——多平台各用各的 key，
        # 必须在单次请求时按平台覆盖 header。
        self.session.headers.update({"Content-Type": "application/json"})

    def _pick_index(self, exclude=None):
        """按 platforms 列表顺序选平台，跳过黑名单与被排除平台。"""
        exclude = exclude or set()
        for i, _ in enumerate(self.platforms):
            if i in exclude or i in ApiClient._dead_platforms:
                continue
            return i
        return None

    def _pick_payload(self, platform, messages, model, max_tokens, temperature):
        return {
            "model": model or platform["model"],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": "none",
        }

    def chat(self, messages, model=None, max_tokens=8192, temperature=0.0, timeout=None):
        if timeout is None:
            timeout = int(os.getenv("API_TIMEOUT", "900"))

        excluded = set()
        last_exc = None
        while True:
            idx = self._pick_index(exclude=excluded)
            if idx is None:
                break
            platform = self.platforms[idx]
            gate = self._gates[idx]
            if gate is not None:
                gate.acquire()
            try:
                payload = self._pick_payload(platform, messages, model,
                                             max_tokens, temperature)
                last_exc = self._call_platform(platform, payload, timeout, idx)
                if isinstance(last_exc, dict):
                    return last_exc
                excluded.add(idx)
            finally:
                if gate is not None:
                    gate.release()
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("所有 API 平台均不可用（额度耗尽或被永久跳过）")

    def _call_platform(self, platform, payload, timeout, idx):
        """单平台内重试；成功返回结果 dict，失败返回异常对象。"""
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(
                    f"{platform['base_url'].rstrip('/')}/chat/completions",
                    json=payload,
                    headers=_platform_headers(platform),
                    timeout=timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                last_exc = e
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt, e)
                    continue
                return e

            if resp.status_code in (401, 403):
                ApiClient._dead_platforms.add(idx)
                return RuntimeError(
                    f"平台{platform['name']} 鉴权失败 {resp.status_code}: {resp.text[:200]}")

            if resp.status_code in self.retry_statuses:
                last_exc = RuntimeError(
                    f"平台{platform['name']} 返回 {resp.status_code}: {resp.text[:200]}")
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt, last_exc)
                    continue
                return last_exc

            try:
                resp.raise_for_status()
            except Exception as e:
                return e

            data = resp.json()
            msg = data["choices"][0]["message"]
            return {
                "content": msg.get("content", ""),
                "reasoning": msg.get("reasoning_content", ""),
                "usage": data.get("usage", {}),
                "platform": platform["name"],
            }
        return last_exc

    def _sleep_before_retry(self, attempt, exc):
        delay = self.retry_backoff * (2 ** attempt)
        print(
            f"  [api] 平台请求第 {attempt + 1} 次失败，{delay:.1f}s 后重试: {exc}",
            flush=True,
        )
        time.sleep(delay)


def get_client() -> ApiClient:
    """返回 ApiClient 单例实例。"""
    return ApiClient()
