# llm_clients.py
# Единые клиенты к GigaChat / Yandex GPT для L (генерация) и B (модерация).
# Убирает дублирование, чинит рекурсию на 401 и добавляет ретрай на невалидный ответ.

from __future__ import annotations

import re
import uuid
import logging
import requests
import urllib3
from typing import Optional, Callable

# GigaChat ходит через verify=False — глушим предупреждения о небезопасном TLS
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


# ==========================================================
# ПАРСИНГ ЧИСЛОВОЙ ОЦЕНКИ РИСКА
# ==========================================================

def parse_risk_value(content: str) -> Optional[float]:
    """
    Извлекает оценку риска 0..1 из ответа LLM.
    Поддерживает: '0.7', '0,7' (рус. запятая), '70%', '7/10', '7 из 10', '1'.
    Возвращает None, если число не найдено.
    """
    if not content:
        return None

    s = content.strip().replace(",", ".")

    # формат "X/10" или "X из 10"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:/|из)\s*(\d+)", s, re.IGNORECASE)
    if m:
        num = float(m.group(1))
        den = float(m.group(2)) or 1.0
        return float(min(1.0, max(0.0, num / den)))

    # формат "70%"
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return float(min(1.0, max(0.0, float(m.group(1)) / 100.0)))

    # обычное число
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        val = float(m.group(1))
        # модель ответила по шкале 0..10 или 0..100 вместо 0..1
        if val > 1.0:
            val = val / 100.0 if val > 10.0 else val / 10.0
        return float(min(1.0, max(0.0, val)))

    return None


def _nonempty(text: str) -> bool:
    return bool(text and text.strip())


# ==========================================================
# GIGACHAT
# ==========================================================

class GigaChatClient:
    TOKEN_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    def __init__(
        self,
        auth_key: str,
        scope: str = "GIGACHAT_API_PERS",
        model: str = "GigaChat",
        timeout: int = 30,
    ):
        if not auth_key:
            raise ValueError("GigaChat auth_key is required")
        self.auth_key = auth_key
        self.scope = scope
        self.model = model
        self.timeout = timeout
        self.access_token: Optional[str] = None
        self.refresh_token()
        logger.info("[GigaChat] client initialized")

    def refresh_token(self):
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {self.auth_key}",
        }
        resp = requests.post(
            self.TOKEN_URL,
            headers=headers,
            data={"scope": self.scope},
            verify=False,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            logger.error(f"[GigaChat] auth failed {resp.status_code}: {resp.text}")
            raise RuntimeError(f"GigaChat auth failed: {resp.status_code}")
        self.access_token = resp.json().get("access_token")
        logger.info("[GigaChat] access token obtained")

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.9,
        max_tokens: int = 100,
        attempts: int = 3,
        validator: Callable[[str], bool] = _nonempty,
    ) -> Optional[str]:
        """
        Возвращает текст ответа, прошедший validator, либо None после attempts попыток.
        401 обновляет токен один раз (без рекурсии).
        """
        if not self.access_token:
            self.refresh_token()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": 1.1,
        }

        token_refreshed = False
        for attempt in range(1, attempts + 1):
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.access_token}",
            }
            try:
                resp = requests.post(
                    self.API_URL,
                    headers=headers,
                    json=payload,
                    verify=False,
                    timeout=self.timeout,
                )

                if resp.status_code == 200:
                    content = (
                        resp.json()
                        .get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        or ""
                    ).strip()
                    if validator(content):
                        return content
                    logger.warning(
                        f"[GigaChat] ответ не прошёл валидацию (попытка {attempt}/{attempts}): '{content}'"
                    )
                    continue

                if resp.status_code == 401 and not token_refreshed:
                    logger.info("[GigaChat] token expired, refreshing...")
                    self.refresh_token()
                    token_refreshed = True
                    continue

                logger.error(f"[GigaChat] API error {resp.status_code}: {resp.text}")

            except Exception as e:
                logger.error(f"[GigaChat] request error (попытка {attempt}/{attempts}): {e}")

        logger.error(f"[GigaChat] не удалось получить валидный ответ за {attempts} попыток")
        return None


# ==========================================================
# YANDEX GPT
# ==========================================================

class YandexGPTClient:
    API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def __init__(
        self,
        folder_id: None,
        api_key: Optional[str] = None,
        iam_token: Optional[str] = None,
        model: str = "yandexgpt-lite",
        timeout: int = 30,
    ):
        if not folder_id:
            raise ValueError("Yandex GPT requires folder_id")
        if not api_key and not iam_token:
            raise ValueError("Yandex GPT requires api_key or iam_token")
        self.folder_id = folder_id
        self.model = model
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Api-Key {api_key}"
        else:
            self.headers["Authorization"] = f"Bearer {iam_token}"
        logger.info("[Yandex GPT] client initialized")

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.9,
        max_tokens: int = 100,
        attempts: int = 3,
        validator: Callable[[str], bool] = _nonempty,
    ) -> Optional[str]:
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
            "messages": [
                {"role": "system", "text": system},
                {"role": "user", "text": user},
            ],
        }

        for attempt in range(1, attempts + 1):
            try:
                resp = requests.post(
                    self.API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if resp.status_code == 200:
                    alts = resp.json().get("result", {}).get("alternatives", [])
                    content = (
                        alts[0].get("message", {}).get("text", "") if alts else ""
                    ).strip()
                    if validator(content):
                        return content
                    logger.warning(
                        f"[Yandex GPT] ответ не прошёл валидацию (попытка {attempt}/{attempts}): '{content}'"
                    )
                    continue

                logger.error(f"[Yandex GPT] API error {resp.status_code}: {resp.text}")

            except Exception as e:
                logger.error(f"[Yandex GPT] request error (попытка {attempt}/{attempts}): {e}")

        logger.error(f"[Yandex GPT] не удалось получить валидный ответ за {attempts} попыток")
        return None
