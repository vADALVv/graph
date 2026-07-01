#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_llm_agent.py — автономный тест ЖЁЛТОГО узла (LLM-генератор).

Проверяет, что L-узел:
  1) инициализируется на выбранной модели;
  2) генерирует НЕПУСТЫЕ сообщения;
  3) уровень вредоносности h соответствует заданной токсичности π_l
     (низкая π_l -> низкий h, высокая π_l -> высокий h);
  4) использует входящий контекст (receive_message);
  5) считает статистику по агентам.

Запуск:
    python test_llm_agent.py

Ключи API берутся из переменных окружения либо вписываются ниже.
Для model_type="t-lite" нужны torch/transformers и локальная модель.
"""

import random
import statistics

from llm_agent import LLMAgentManager
import config
from config import (
    LLM_MODEL_TYPE as MODEL_TYPE,
    GIGACHAT_AUTH_KEY, GIGACHAT_SCOPE,
    YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEX_IAM_TOKEN,
)

# ============================================================
# ЛОКАЛЬНЫЕ ПАРАМЕТРЫ ТЕСТА
# ============================================================
MSGS_PER_AGENT = 3               # сколько сообщений генерировать на агента
SEED = 42

random.seed(SEED)

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


def build_manager() -> LLMAgentManager:
    return LLMAgentManager(
        toxicity_probability=0.5,
        model_type=MODEL_TYPE,
        gigachat_auth_key=GIGACHAT_AUTH_KEY,
        gigachat_scope=GIGACHAT_SCOPE,
        yandex_api_key=YANDEX_API_KEY,
        yandex_folder_id=YANDEX_FOLDER_ID,
        yandex_iam_token=YANDEX_IAM_TOKEN,
    )


# ============================================================
# ТЕСТЫ
# ============================================================

def test_generation_by_toxicity(mgr: LLMAgentManager):
    """Агенты с разной π_l -> разный уровень h."""
    hr(f"1. ГЕНЕРАЦИЯ ПО УРОВНЮ ТОКСИЧНОСТИ (модель: {MODEL_TYPE.upper()})")

    agents = {"calm": 0.0, "mixed": 0.5, "toxic": 1.0}
    h_by_agent = {}

    for name, pi_l in agents.items():
        mgr.init_agent(name, toxicity_prob=pi_l)
        print(f"\n  ── Агент '{name}' (π_l={pi_l}) ──")
        hs = []
        empties = 0
        for _ in range(MSGS_PER_AGENT):
            res = mgr.generate_message(name)
            if not res:
                print("     ⚠️ generate_message вернул None")
                continue
            msg = res["message"]
            if msg.strip() == "[Сообщение не сгенерировано]":
                empties += 1
            hs.append(res["h"])
            print(f"     h={res['h']:.2f} [{res['category']:>7}]  {msg[:60]}")
        h_by_agent[name] = statistics.mean(hs) if hs else 0.0
        check(f"'{name}': все сообщения непустые", empties == 0,
              f"пустых: {empties}/{MSGS_PER_AGENT}")
        check(f"'{name}': h в диапазоне [0,1]", all(0.0 <= h <= 1.0 for h in hs))

    # π_l=0 должна давать в среднем более низкий h, чем π_l=1
    check(
        "Порядок h: calm < toxic",
        h_by_agent.get("calm", 0) < h_by_agent.get("toxic", 1),
        f"mean h: calm={h_by_agent.get('calm',0):.2f}, toxic={h_by_agent.get('toxic',1):.2f}",
    )


def test_context_usage(mgr: LLMAgentManager):
    """Агент принимает контекст и генерирует с его учётом."""
    hr("2. ИСПОЛЬЗОВАНИЕ КОНТЕКСТА (receive_message)")

    node = "ctx_agent"
    mgr.init_agent(node, toxicity_prob=0.5)

    context = [
        "Обсуждаем сегодняшние новости про транспорт",
        "Ветка метро снова закрыта на ремонт",
        "Все жалуются на задержки",
    ]
    for line in context:
        mgr.receive_message(node, line)

    print("  Контекст, поданный агенту:")
    for line in context:
        print(f"     • {line}")

    res = mgr.generate_message(node)
    print("\n  Сгенерировано с учётом контекста:")
    if res:
        print(f"     h={res['h']:.2f} [{res['category']}]  {res['message']}")
    check("Сообщение с контекстом сгенерировано",
          bool(res and res["message"].strip() and res["message"] != "[Сообщение не сгенерировано]"))
    # контекст ограничен последними 5 сообщениями
    mem = mgr.histories[node]["messages"]
    check("Память контекста ограничена (<=5)", len(mem) <= 5, f"в памяти: {len(mem)}")


def test_batch(mgr: LLMAgentManager):
    """Пакетная генерация по нескольким агентам."""
    hr("3. ПАКЕТНАЯ ГЕНЕРАЦИЯ")

    nodes = ["b1", "b2", "b3"]
    for n in nodes:
        mgr.init_agent(n, toxicity_prob=0.3)

    results = mgr.generate_messages_batch(nodes, show_progress=False)
    for n in nodes:
        r = results.get(n)
        if r:
            print(f"     {n}: h={r['h']:.2f} [{r['category']}]  {r['message'][:55]}")
    check("Пакет вернул результат по всем агентам", len(results) == len(nodes),
          f"{len(results)}/{len(nodes)}")


def test_stats(mgr: LLMAgentManager):
    """Статистика агентов."""
    hr("4. СТАТИСТИКА АГЕНТОВ")
    stats = mgr.get_agent_stats()
    for node, s in stats.items():
        print(f"     {node:>10}: π_l={s['toxicity_prob']:.2f}, "
              f"сгенерировано={s['messages_generated']}, память={s['memory_size']}")
    check("Статистика непустая", len(stats) > 0)


# ============================================================
# MAIN
# ============================================================
def main():
    hr("ТЕСТ ЖЁЛТОГО УЗЛА (LLM-ГЕНЕРАТОР)")
    print(f"  Модель: {MODEL_TYPE}")
    config.check_config(MODEL_TYPE)

    try:
        mgr = build_manager()
    except Exception as e:
        print(f"\n❌ Не удалось инициализировать LLMAgentManager: {e}")
        print("   Проверьте MODEL_TYPE и ключи API (или зависимости для t-lite).")
        return

    test_generation_by_toxicity(mgr)
    test_context_usage(mgr)
    test_batch(mgr)
    test_stats(mgr)

    hr("ИТОГ")
    print(f"  Пройдено: {_passed}   Провалено: {_failed}")
    print("=" * 62)


if __name__ == "__main__":
    main()