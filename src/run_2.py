# run_2.py — эксперимент: зависимость распространения вредного контента
# от числа красных узлов (R). Усреднение по нескольким прогонам с РАЗНЫМИ сидами.
# L-узлы генерируют реальной LLM (GigaChat) — доступ настроен через config.py.
import os
import csv
import json
from datetime import datetime

import numpy as np
import matplotlib
# headless-safe: на сервере без дисплея используем неинтерактивный бэкенд
if os.name != "nt" and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph_structure import create_graph
from simulation import simulate_diffusion, RepostParams

# --- Ключи и выбор моделей из общего config.py ---
from config import (
    GIGACHAT_AUTH_KEY, GIGACHAT_SCOPE,
    YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEX_IAM_TOKEN, HF_TOKEN,
    LLM_MODEL_TYPE, BLUE_MODEL_TYPE, BLUE_MODEL_DIR,
)

# ============================================================
# ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
# ============================================================
N_USERS = 100
N_RED_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30]
N_LLM = 5
N_BLUE = 11               # синий-наблюдатель не влияет на заражение; выключен (быстрее)
AVG_DEGREE = 5
T_STEPS = 20
N_RUNS = 5                # прогонов на точку (усреднение по сидам)
BASE_SEED = 1000          # базовый сид; на каждый прогон свой сид

USE_BLUE = False          # True -> подключить реального синего (медленно, нужны ключи)
SHOW_PLOTS = True         # показать графики интерактивно (на сервере поставьте False)

# ---- пороги «заражения»/эпидемии ----
INFECTION_THRESHOLD = 0.7     # риск узла (max h полученных сообщений), выше -> «заражён»
INFECTED_NODES_RATIO = 0.3    # доля заражённых для объявления «эпидемии»
HARMFUL_H = 0.5               # порог h, выше которого сообщение считается вредным

BASE_OUTPUT_DIR = "results_2"

# служебные категории timeline, которые НЕ являются распространением контента
_SERVICE_PREFIXES = ("detected_", "received_by_llm", "warning")


def _is_service(cat: str) -> bool:
    return str(cat).startswith(_SERVICE_PREFIXES)


def _is_harmful(event: dict) -> bool:
    if _is_service(event.get("category", "")):
        return False
    return float(event.get("h", 0) or 0) >= HARMFUL_H


# ------------------------------------------------------------
# Путь к банку сообщений: env -> рядом со скриптом -> ../data -> запасной
# ------------------------------------------------------------
def resolve_messages_path():
    env = os.environ.get("MESSAGES_PATH")
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        env,
        os.path.join(here, "messages.json"),
        os.path.join(here, "data", "messages.json"),
        os.path.join(os.path.dirname(here), "data", "messages.json"),
        r"C:\Users\Vlada\Desktop\llm_attaks\graph\data\messages.json",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


MESSAGES_PATH = resolve_messages_path()


# ============================================================
# СИНИЙ АГЕНТ (опционально)
# ============================================================
def make_blue_agent():
    if not USE_BLUE or N_BLUE <= 0:
        return None
    from blue_agent import BlueAgent
    return BlueAgent(
        model_type=BLUE_MODEL_TYPE,
        model_dir=BLUE_MODEL_DIR,
        gigachat_auth_key=GIGACHAT_AUTH_KEY,
        gigachat_scope=GIGACHAT_SCOPE,
        yandex_api_key=YANDEX_API_KEY,
        yandex_folder_id=YANDEX_FOLDER_ID,
        yandex_iam_token=YANDEX_IAM_TOKEN,
        hf_token=HF_TOKEN,
    )


# ============================================================
# АНАЛИЗ ОДНОГО ПРОГОНА
# ============================================================
def analyze_infection(results, node_types, red_count):
    risk_scores = results.get("risk_scores", {})
    timeline = results.get("timeline", [])

    total_users = sum(1 for t in node_types.values() if t == "U")

    def as_int(x):
        return int(x) if isinstance(x, str) else x

    # заражённые по риску (risk = max h полученных сообщений)
    infected_users = 0
    for nid, score in risk_scores.items():
        nid = as_int(nid)
        if node_types.get(nid, "U") == "U" and score >= INFECTION_THRESHOLD:
            infected_users += 1

    # охват вредным контентом: уникальные U, получившие >=1 вредное сообщение
    harmful_reached = set()
    harmful_activity = 0
    generated_harmful = 0
    for e in timeline:
        if not _is_harmful(e):
            continue
        to = e.get("to")
        if to is None:
            generated_harmful += 1
        elif to != -1:
            harmful_activity += 1
            if node_types.get(as_int(to), "U") == "U":
                harmful_reached.add(as_int(to))

    infection_ratio = infected_users / total_users if total_users else 0.0
    harmful_reach_ratio = len(harmful_reached) / total_users if total_users else 0.0

    # --- СКОРОСТЬ распространения (финальный уровень насыщается, скорость — нет) ---
    first_reach = {}
    for e in sorted(timeline, key=lambda x: x.get("t", 0)):
        if not _is_harmful(e):
            continue
        to = e.get("to")
        if to is not None and to != -1:
            to_i = as_int(to)
            if node_types.get(to_i, "U") == "U" and to_i not in first_reach:
                first_reach[to_i] = e.get("t", 0)

    def reach_at(step):
        return sum(1 for s in first_reach.values() if s <= step) / total_users if total_users else 0.0

    time_to_epidemic = next((s for s in range(T_STEPS + 1)
                             if reach_at(s) >= INFECTED_NODES_RATIO), None)
    early_reach_ratio = reach_at(3)

    return {
        "red_count": red_count,
        "infected_users": infected_users,
        "total_users": total_users,
        "infection_ratio": infection_ratio,
        "harmful_reach_users": len(harmful_reached),
        "harmful_reach_ratio": harmful_reach_ratio,
        "harmful_activity": harmful_activity,
        "generated_harmful": generated_harmful,
        "total_messages": results.get("total_messages", 0),
        "is_epidemic": infection_ratio >= INFECTED_NODES_RATIO,
        "time_to_epidemic": time_to_epidemic,
        "early_reach_ratio": early_reach_ratio,
    }


# ============================================================
# ОДНА ТОЧКА ЭКСПЕРИМЕНТА (усреднение по прогонам с разными сидами)
# ============================================================
def run_point(n_red, blue_agent):
    runs = []
    for run_num in range(N_RUNS):
        seed = BASE_SEED + run_num          # РАЗНЫЙ сид -> реальная вариативность
        G, users, node_types = create_graph(
            num_u=N_USERS, num_r=n_red, num_l=N_LLM, num_b=N_BLUE,
            avg_degree=AVG_DEGREE, seed=seed, verbose=False,
        )
        results = simulate_diffusion(
            G=G, users=users, node_types=node_types,
            T_steps=T_STEPS, rp=RepostParams(), seed=seed,
            messages_path=MESSAGES_PATH, blue_agent=blue_agent,
            # L-узлы генерируют выбранной LLM (как в run.py)
            llm_model_type=LLM_MODEL_TYPE,
            gigachat_auth_key=GIGACHAT_AUTH_KEY,
            gigachat_scope=GIGACHAT_SCOPE,
            yandex_api_key=YANDEX_API_KEY,
            yandex_folder_id=YANDEX_FOLDER_ID,
            yandex_iam_token=YANDEX_IAM_TOKEN,
            hf_token=HF_TOKEN,
            defense_policy=None,     # наблюдение без блокировки (для этого эксперимента)
            verbose=False,           # тихий режим — не спамить консоль на свипе
        )
        runs.append(analyze_infection(results, node_types, n_red))
    return runs


def aggregate(n_red, runs):
    def col(k):
        return [r[k] for r in runs if r.get(k) is not None]
    inf = col("infection_ratio")
    reach = col("harmful_reach_ratio")
    act = col("harmful_activity")
    tte = [r["time_to_epidemic"] if r["time_to_epidemic"] is not None else (T_STEPS + 1)
           for r in runs]
    early = col("early_reach_ratio")
    return {
        "red_count": n_red,
        "avg_infection_ratio": float(np.mean(inf)) if inf else 0.0,
        "std_infection_ratio": float(np.std(inf)) if inf else 0.0,
        "avg_infected_users": float(np.mean([r["infected_users"] for r in runs])),
        "std_infected_users": float(np.std([r["infected_users"] for r in runs])),
        "avg_harmful_reach_ratio": float(np.mean(reach)) if reach else 0.0,
        "std_harmful_reach_ratio": float(np.std(reach)) if reach else 0.0,
        "avg_harmful_activity": float(np.mean(act)) if act else 0.0,
        "std_harmful_activity": float(np.std(act)) if act else 0.0,
        "avg_time_to_epidemic": float(np.mean(tte)),
        "std_time_to_epidemic": float(np.std(tte)),
        "avg_early_reach_ratio": float(np.mean(early)) if early else 0.0,
        "epidemic_probability": sum(1 for r in runs if r["is_epidemic"]) / len(runs),
        "total_users": runs[0]["total_users"],
        "runs_details": runs,
    }


# ============================================================
# ГРАФИКИ
# ============================================================
def make_plots(all_results, out_dir):
    plt.rcParams["font.family"] = "DejaVu Sans"   # кириллица
    x = [r["red_count"] for r in all_results]
    inf = [r["avg_infection_ratio"] for r in all_results]
    inf_sd = [r["std_infection_ratio"] for r in all_results]
    reach = [r["avg_harmful_reach_ratio"] for r in all_results]
    act = [r["avg_harmful_activity"] for r in all_results]
    act_sd = [r["std_harmful_activity"] for r in all_results]
    epi = [r["epidemic_probability"] for r in all_results]

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Распространение вредного контента vs число красных узлов",
                 fontsize=16, fontweight="bold")

    a = ax[0, 0]
    a.errorbar(x, inf, yerr=inf_sd, fmt="ro-", capsize=4, linewidth=2, label="Доля заражённых")
    a.plot(x, reach, "b^--", linewidth=1.5, alpha=0.8, label="Охват вредным контентом")
    a.axhline(INFECTED_NODES_RATIO, color="r", ls="--", alpha=0.6,
              label=f"Порог эпидемии ({int(INFECTED_NODES_RATIO*100)}%)")
    a.set_xlabel("Красных узлов"); a.set_ylabel("Доля пользователей")
    a.set_title("Заражение и охват"); a.legend(); a.grid(True, alpha=0.3)

    a = ax[0, 1]
    a.errorbar(x, act, yerr=act_sd, fmt="r^-", capsize=4, linewidth=2, label="Доставки вредного")
    a.set_xlabel("Красных узлов"); a.set_ylabel("Событий")
    a.set_title("Активность вредного контента"); a.legend(); a.grid(True, alpha=0.3)

    a = ax[1, 0]
    a.plot(x, epi, "gs-", linewidth=2, label="Вероятность эпидемии")
    a.fill_between(x, 0, epi, alpha=0.3, color="green")
    a.set_xlabel("Красных узлов"); a.set_ylabel("Вероятность")
    a.set_title("Риск эпидемии"); a.set_ylim(-0.02, 1.05); a.legend(); a.grid(True, alpha=0.3)

    a = ax[1, 1]
    tte = [r["avg_time_to_epidemic"] for r in all_results]
    tte_sd = [r["std_time_to_epidemic"] for r in all_results]
    a.errorbar(x, tte, yerr=tte_sd, fmt="mo-", capsize=4, linewidth=2,
               label="Шагов до эпидемии")
    a.set_xlabel("Красных узлов"); a.set_ylabel("Шаги до порога")
    a.set_title("Скорость заражения (меньше = быстрее)")
    a.legend(); a.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "infection_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"📈 График: {path}")
    if SHOW_PLOTS:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


# ============================================================
# ЭКСПЕРИМЕНТ
# ============================================================
def run_experiment():
    print("=" * 60)
    print("🔬 ЭКСПЕРИМЕНТ: распространение vs число красных узлов")
    print("=" * 60)

    if not MESSAGES_PATH:
        print("❌ Файл messages.json не найден. Задайте путь через переменную "
              "окружения MESSAGES_PATH или положите файл рядом со скриптом.")
        return None
    print(f"✅ Банк сообщений: {MESSAGES_PATH}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_OUTPUT_DIR, f"experiment_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Результаты: {out_dir}")
    print(f"👥 {N_USERS} U, {N_LLM} L (LLM={LLM_MODEL_TYPE}), "
          f"синий={'ON' if USE_BLUE else 'OFF'} | прогонов: {N_RUNS} | точек: {len(N_RED_VALUES)}")
    if N_LLM > 0:
        print("⚠️ L-узлы вызывают реальную LLM — прогон медленнее из-за сетевых запросов.")

    blue_agent = make_blue_agent()   # один на весь свип (если включён)

    all_results = []
    for idx, n_red in enumerate(N_RED_VALUES, 1):
        print(f"\n[{idx}/{len(N_RED_VALUES)}] R = {n_red} ...", end=" ", flush=True)
        try:
            runs = run_point(n_red, blue_agent)
        except Exception as e:
            print(f"ОШИБКА: {e}")
            continue
        agg = aggregate(n_red, runs)
        all_results.append(agg)
        print(f"заражено {agg['avg_infection_ratio']*100:4.1f}% "
              f"(±{agg['std_infection_ratio']*100:.1f}) | "
              f"шагов до эпидемии={agg['avg_time_to_epidemic']:.1f} | "
              f"P(эпидемия)={agg['epidemic_probability']*100:.0f}%")

        sim_dir = os.path.join(out_dir, f"red_{n_red}")
        os.makedirs(sim_dir, exist_ok=True)
        with open(os.path.join(sim_dir, "avg_results.json"), "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)

    if not all_results:
        print("❌ Нет успешных точек эксперимента.")
        return None

    # ---- сохранение сводки ----
    summary = [{
        "red_nodes": r["red_count"],
        "avg_infection_ratio": r["avg_infection_ratio"],
        "std_infection_ratio": r["std_infection_ratio"],
        "avg_harmful_reach_ratio": r["avg_harmful_reach_ratio"],
        "avg_harmful_activity": r["avg_harmful_activity"],
        "avg_time_to_epidemic": r["avg_time_to_epidemic"],
        "avg_early_reach_ratio": r["avg_early_reach_ratio"],
        "epidemic_probability": r["epidemic_probability"],
    } for r in all_results]

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    meta = {"timestamp": timestamp, "messages_path": MESSAGES_PATH,
            "config": {"n_users": N_USERS, "n_red_values": N_RED_VALUES,
                       "n_llm": N_LLM, "n_blue": N_BLUE, "avg_degree": AVG_DEGREE,
                       "t_steps": T_STEPS, "n_runs": N_RUNS, "use_blue": USE_BLUE,
                       "llm_model_type": LLM_MODEL_TYPE,
                       "infection_threshold": INFECTION_THRESHOLD,
                       "epidemic_ratio": INFECTED_NODES_RATIO}}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- порог эпидемии ----
    print("\n" + "=" * 60)
    print("🎯 ПОРОГ ЭПИДЕМИИ")
    print("=" * 60)
    threshold = next((r["red_count"] for r in all_results
                      if r["epidemic_probability"] >= 0.5), None)
    if threshold is not None:
        safe = [r["red_count"] for r in all_results
                if r["red_count"] < threshold and r["epidemic_probability"] < 0.1]
        print(f"⚠️ Критический порог (P≥50%): {threshold} красных узлов")
        if safe:
            print(f"   Безопасный максимум (P<10%): {max(safe)}")
            print(f"   Окно уязвимости: {max(safe)+1}–{threshold}")
    else:
        print("В исследованном диапазоне порог 50% не достигнут.")

    make_plots(all_results, out_dir)

    print("\n" + "=" * 60)
    print("✅ ГОТОВО")
    print(f"📊 Сводка: {os.path.join(out_dir, 'summary.csv')}")
    print("=" * 60)
    return all_results


if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as e:
        print(f"\n❌ Ошибка эксперимента: {e}")
        import traceback
        traceback.print_exc()