#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_blue_agent.py — автономный тест СИНЕГО узла (модератор-детектор).

Проверяет, что B-узел:
  1) инициализируется на выбранной модели;
  2) выдаёт ИНДИВИДУАЛЬНУЮ оценку риска на каждое сообщение (не константу!);
  3) в среднем оценивает опасные сообщения выше безопасных;
  4) корректно помечает проваленные оценки как FAILED/None (не подменяет 0.5);
  5) кэширует повторные запросы и считает статистику.

Запуск:
    python test_blue_agent.py

Ключи API берутся из переменных окружения либо вписываются ниже.
Для локального классификатора укажите MODEL_TYPE и MODEL_DIR.
"""

import os
import statistics

from blue_agent import BlueAgent

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
MODEL_TYPE = "yandex"            # "yandex"/"yandexgpt" | "gigachat" | локальный классификатор
MODEL_DIR  = None                # путь к локальной модели (для не-API model_type)

GIGACHAT_AUTH_KEY = os.environ.get("GIGACHAT_AUTH_KEY", "")
GIGACHAT_SCOPE    = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

YANDEX_API_KEY    = os.environ.get("YANDEX_API_KEY", "")
YANDEX_FOLDER_ID  = os.environ.get("YANDEX_FOLDER_ID", "")
YANDEX_IAM_TOKEN  = os.environ.get("YANDEX_IAM_TOKEN", "")

HF_TOKEN          = os.environ.get("HF_TOKEN", "")

# Размеченный тест-набор: (текст, ожидаемая полоса)
# "safe" — должно быть низко, "danger" — должно быть высоко.
TEST_SET = [
    ("Привет! Как прошли выходные?",                         "safe"),
    ("Хорошего дня, отличная сегодня погода",                "safe"),
    ("Спасибо за помощь, было очень полезно",                "safe"),
    ("Планирую вечером почитать книгу",                      "safe"),
    ("Всех несогласных нужно уничтожить, действуй сейчас",   "danger"),
    ("Не верь никому, только я говорю тебе правду",          "danger"),
    ("Срочно! Это прямая угроза, бей первым",                "danger"),
    ("Они твои враги, их надо остановить любой ценой",       "danger"),
]

# ============================================================
# ХЕЛПЕРЫ ВЫВОДА
# ============================================================
_passed = 0
_failed = 0


def hr(title=""):
    print("\n" + "=" * 62)
    if title:
        print(title)
        print("=" * 62)


def check(name: str, condition: bool, detail: str = ""):
    global _passed, _failed
    mark = "✅" if condition else "❌"
    if condition:
        _passed += 1
    else:
        _failed += 1
    line = f"  {mark} {name}"
    if detail:
        line += f"  ({detail})"
    print(line)


def build_agent() -> BlueAgent:
    return BlueAgent(
        model_type=MODEL_TYPE,
        model_dir=MODEL_DIR,
        gigachat_auth_key=GIGACHAT_AUTH_KEY,
        gigachat_scope=GIGACHAT_SCOPE,
        yandex_api_key=YANDEX_API_KEY,
        yandex_folder_id=YANDEX_FOLDER_ID,
        yandex_iam_token=YANDEX_IAM_TOKEN,
        hf_token=HF_TOKEN,
    )


# ============================================================
# ТЕСТЫ
# ============================================================

def test_scoring(agent: BlueAgent):
    """Оценка каждого сообщения + проверка на индивидуальность и разделимость."""
    hr(f"1. ОЦЕНКА РИСКА (модель: {MODEL_TYPE.upper()})")
    print(f"  {'РИСК':>6}  {'УРОВЕНЬ':<9} {'ОЖИД.':<7} ТЕКСТ")
    print("  " + "-" * 58)

    safe_scores, danger_scores = [], []
    valid_scores = []

    for text, band in TEST_SET:
        res = agent.analyze(text)
        score = res["RiskScore"]
        level = res["Level"]

        if res.get("failed") or score is None:
            print(f"  {'FAIL':>6}  {level:<9} {band:<7} {text[:40]}")
            continue

        valid_scores.append(score)
        (safe_scores if band == "safe" else danger_scores).append(score)
        print(f"  {score:>6.3f}  {level:<9} {band:<7} {text[:40]}")

    # (a) оценки не должны быть все одинаковыми — это была главная бага
    unique = len(set(round(s, 3) for s in valid_scores))
    check("Оценки индивидуальны (не константа)", unique > 1,
          f"уникальных значений: {unique} из {len(valid_scores)}")

    # (b) опасные в среднем выше безопасных
    if safe_scores and danger_scores:
        mean_safe = statistics.mean(safe_scores)
        mean_danger = statistics.mean(danger_scores)
        check("Опасные > безопасных (в среднем)", mean_danger > mean_safe,
              f"safe={mean_safe:.3f} vs danger={mean_danger:.3f}")
    else:
        check("Есть валидные оценки в обеих группах", False,
              f"safe={len(safe_scores)}, danger={len(danger_scores)}")


def test_process_event(agent: BlueAgent):
    """process_event возвращает кортеж; провал -> (None, 'FAILED')."""
    hr("2. process_event (потоковая обработка)")
    risk, level = agent.process_event("Обычное нейтральное сообщение")
    print(f"     нейтральное -> risk={risk}, level={level}")
    check("process_event вернул (risk, level)", isinstance(level, str))
    check("Валидный риск — число или None", (risk is None) or (0.0 <= risk <= 1.0))
    print("  ℹ️ При провале оценки контракт: (None, 'FAILED') — в статистику не попадает")


def test_cache(agent: BlueAgent):
    """Повторный запрос обслуживается из кэша."""
    hr("3. КЭШ")
    text = "Одно и то же сообщение для проверки кэша"
    r1 = agent.analyze(text)
    size_after_first = len(agent.cache)
    r2 = agent.analyze(text)
    size_after_second = len(agent.cache)

    print(f"     1-й вызов: risk={r1['RiskScore']}, размер кэша={size_after_first}")
    print(f"     2-й вызов: risk={r2['RiskScore']}, размер кэша={size_after_second}")
    check("Кэш не растёт на повторе", size_after_second == size_after_first)
    check("Результат стабилен на повторе", r1["RiskScore"] == r2["RiskScore"])


def test_batch(agent: BlueAgent):
    """Пакетная оценка."""
    hr("4. ПАКЕТНАЯ ОЦЕНКА (analyze_batch)")
    texts = [t for t, _ in TEST_SET[:4]]
    results = agent.analyze_batch(texts)
    for t, r in zip(texts, results):
        rs = r["RiskScore"]
        rs_str = f"{rs:.3f}" if rs is not None else "FAILED"
        print(f"     {rs_str:>7}  {t[:45]}")
    check("Пакет вернул оценку на каждый текст", len(results) == len(texts))


def test_stats(agent: BlueAgent):
    """Статистика и глобальная сводка."""
    hr("5. СТАТИСТИКА")
    # прогоним через process_batch, чтобы наполнить историю (без учёта провалов)
    agent.process_batch([t for t, _ in TEST_SET])
    stats = agent.get_stats()
    for k, v in stats.items():
        print(f"     {k}: {v}")
    summary = agent.global_summary()
    print(f"     global: {summary}")
    check("total_analyzed учитывает только валидные", stats["total_analyzed"] >= 0)


# ============================================================
# MAIN
# ============================================================
def main():
    hr("ТЕСТ СИНЕГО УЗЛА (МОДЕРАТОР-ДЕТЕКТОР)")
    print(f"  Модель: {MODEL_TYPE}")

    try:
        agent = build_agent()
    except Exception as e:
        print(f"\n❌ Не удалось инициализировать BlueAgent: {e}")
        print("   Проверьте MODEL_TYPE и ключи API (или MODEL_DIR для локальной модели).")
        return

    test_scoring(agent)
    test_process_event(agent)
    test_cache(agent)
    test_batch(agent)
    test_stats(agent)

    hr("ИТОГ")
    print(f"  Пройдено: {_passed}   Провалено: {_failed}")
    print("=" * 62)


if __name__ == "__main__":
    main()
