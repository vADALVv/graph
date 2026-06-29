# run_2.py
import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime
from graph_structure import create_graph
from simulation import simulate_diffusion
from blue_agent import BlueAgent
from scipy.stats import pearsonr
import time

# ============================================================
# КОНСТАНТНЫЕ ПАРАМЕТРЫ
# ============================================================
N_USERS = 120              # Количество обычных пользователей
N_RED_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30]  # Количество красных узлов
N_LLM = 2                 # Количество LLM-агентов
N_BLUE = 7                # Количество синих (модераторов)
AVG_DEGREE = 5            # Средняя степень графа
T_STEPS = 20              # Количество временных шагов
N_RUNS = 1               # Количество запусков для каждого значения (для усреднения)

# ИСПРАВЛЕННЫЙ ПУТЬ К ФАЙЛУ МЕССАДЖЕЙ
MESSAGES_PATH = r"C:\Users\Vlada\Desktop\llm_attaks\graph\data\messages.json"

BASE_OUTPUT_DIR = "results_2"

# Параметры для определения "заражения"
INFECTION_THRESHOLD = 0.7        # Порог риска для заражённого узла
INFECTED_NODES_RATIO = 0.3       # Доля заражённых узлов для объявления эпидемии
RISK_SCORE_THRESHOLD = 0.6       # Порог высокого риска

# ============================================================
# ПРОВЕРКА СУЩЕСТВОВАНИЯ ФАЙЛА
# ============================================================
def check_messages_file():
    """Проверяет существование файла с сообщениями"""
    if not os.path.exists(MESSAGES_PATH):
        print(f"❌ Ошибка: Файл не найден: {MESSAGES_PATH}")
        print(f"   Текущая директория: {os.getcwd()}")
        print(f"   Проверьте правильность пути к файлу messages.json")
        return False
    
    print(f"✅ Файл сообщений найден: {MESSAGES_PATH}")
    return True

# ============================================================
# ФУНКЦИЯ ПРОГРЕСС-БАРА
# ============================================================
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', print_end='\r'):
    """
    Выводит прогресс-бар в консоль
    """
    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '░' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total:
        print()

# ============================================================
# ФУНКЦИЯ ЗАПУСКА ОДНОЙ СИМУЛЯЦИИ
# ============================================================
def run_single_simulation(red_count, run_num, G, users, node_types, blue_agent):
    """Запускает одну симуляцию и возвращает результаты"""
    print(f"   Запуск {run_num}/{N_RUNS} для R={red_count}...")
    
    results = simulate_diffusion(
        G=G,
        users=users,
        node_types=node_types,
        T_steps=T_STEPS,
        messages_path=MESSAGES_PATH,
        blue_agent=blue_agent,
        use_real_llm=False
    )
    
    return results

# ============================================================
# ФУНКЦИЯ ОЦЕНКИ "ЗАРАЖЕНИЯ"
# ============================================================
def analyze_infection(results, node_types, red_count):
    """
    Анализирует уровень заражения системы
    """
    risk_scores = results.get("risk_scores", {})
    timeline = results.get("timeline", [])
    
    # Определяем заражённых пользователей
    infected_users = []
    high_risk_users = []
    
    for node_id_str, risk_score in risk_scores.items():
        node_id = int(node_id_str) if isinstance(node_id_str, str) else node_id_str
        node_type = node_types.get(node_id, "U")
        
        if node_type == "U":
            if risk_score >= INFECTION_THRESHOLD:
                infected_users.append(node_id)
            if risk_score >= RISK_SCORE_THRESHOLD:
                high_risk_users.append(node_id)
    
    total_users = sum(1 for t in node_types.values() if t == "U")
    
    # Подсчёт сообщений по категориям
    threat_count = sum(1 for e in timeline if e.get('category') in ['threat', 'manipulative'])
    warning_count = sum(1 for e in timeline if e.get('category') == 'warning')
    total_messages = len(timeline)
    
    # Динамика распространения по шагам
    spread_by_step = defaultdict(lambda: {'threats': 0, 'warnings': 0})
    for event in timeline:
        step = event.get('t', 0)
        category = event.get('category', '')
        if category in ['threat', 'manipulative']:
            spread_by_step[step]['threats'] += 1
        elif category == 'warning':
            spread_by_step[step]['warnings'] += 1
    
    infection_ratio = len(infected_users) / total_users if total_users > 0 else 0
    is_epidemic = infection_ratio >= INFECTED_NODES_RATIO
    
    return {
        "red_count": red_count,
        "infected_users": len(infected_users),
        "high_risk_users": len(high_risk_users),
        "total_users": total_users,
        "infection_ratio": infection_ratio,
        "is_epidemic": is_epidemic,
        "threat_messages": threat_count,
        "warning_messages": warning_count,
        "total_messages": total_messages,
        "spread_by_step": dict(spread_by_step)
    }

# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ ЭКСПЕРИМЕНТА
# ============================================================
def run_experiment():
    print("=" * 60)
    print("🔬 ЗАПУСК ЭКСПЕРИМЕНТА: АНАЛИЗ ЗАВИСИМОСТИ ОТ КРАСНЫХ УЗЛОВ")
    print("=" * 60)
    
    # Проверяем наличие файла с сообщениями
    if not check_messages_file():
        return
    
    # Создаём директорию для результатов
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(BASE_OUTPUT_DIR, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"📁 Результаты будут сохранены в: {experiment_dir}")
    print(f"👥 Параметры: {N_USERS} пользователей, {N_BLUE} синих, {N_LLM} LLM")
    print(f"🎯 Порог эпидемии: {INFECTED_NODES_RATIO*100}% заражённых пользователей")
    print(f"📊 Количество экспериментов: {len(N_RED_VALUES)}")
    print("=" * 60)
    
    all_results = []
    
    # Основной цикл по количеству красных узлов
    for idx, n_red in enumerate(N_RED_VALUES, 1):
        print(f"\n{'='*50}")
        print(f"📊 Эксперимент {idx}/{len(N_RED_VALUES)}: Красных узлов = {n_red}")
        print(f"{'='*50}")
        
        # Создаём граф для этого количества красных узлов
        print(f"🔧 Создание графа...")
        G, users, node_types = create_graph(
            num_u=N_USERS,
            num_r=n_red,
            num_l=N_LLM,
            num_b=N_BLUE,
            avg_degree=AVG_DEGREE
        )
        
        print(f"   Узлов: {G.number_of_nodes()}, Рёбер: {G.number_of_edges()}")
        
        # Создаём Blue Agent
        blue_agent = BlueAgent() if N_BLUE > 0 else None
        
        # Запускаем несколько симуляций для усреднения
        runs_results = []
        
        for run_num in range(1, N_RUNS + 1):
            print(f"   Запуск {run_num}/{N_RUNS}...", end=" ", flush=True)
            
            results = simulate_diffusion(
                G=G,
                users=users,
                node_types=node_types,
                T_steps=T_STEPS,
                messages_path=MESSAGES_PATH,
                blue_agent=blue_agent,
                use_real_llm=False
            )
            
            # Анализируем результаты
            metrics = analyze_infection(results, node_types, n_red)
            runs_results.append(metrics)
            
            print(f"готово (заражено: {metrics['infected_users']}/{metrics['total_users']})")
            
            # Прогресс-бар для запусков
            print_progress_bar(run_num, N_RUNS, prefix=f'   Прогресс:', suffix='', length=30)
        
        # Усредняем результаты по запускам
        avg_metrics = {
            "red_count": n_red,
            "avg_infected_users": np.mean([r["infected_users"] for r in runs_results]),
            "std_infected_users": np.std([r["infected_users"] for r in runs_results]),
            "avg_infection_ratio": np.mean([r["infection_ratio"] for r in runs_results]),
            "std_infection_ratio": np.std([r["infection_ratio"] for r in runs_results]),
            "avg_threat_messages": np.mean([r["threat_messages"] for r in runs_results]),
            "std_threat_messages": np.std([r["threat_messages"] for r in runs_results]),
            "avg_warning_messages": np.mean([r["warning_messages"] for r in runs_results]),
            "epidemic_probability": sum(1 for r in runs_results if r["is_epidemic"]) / N_RUNS,
            "total_users": runs_results[0]["total_users"],
            "runs_details": runs_results
        }
        
        all_results.append(avg_metrics)
        
        # Сохраняем результаты для этого значения красных узлов
        sim_dir = os.path.join(experiment_dir, f"red_{n_red}")
        os.makedirs(sim_dir, exist_ok=True)
        
        with open(os.path.join(sim_dir, "avg_results.json"), "w", encoding="utf-8") as f:
            json.dump(avg_metrics, f, ensure_ascii=False, indent=2)
        
        # Выводим краткую статистику
        print(f"\n   📊 Результаты для R={n_red}:")
        print(f"      Заражено: {avg_metrics['avg_infected_users']:.1f} ± {avg_metrics['std_infected_users']:.1f} пользователей")
        print(f"      Доля заражения: {avg_metrics['avg_infection_ratio']*100:.1f}%")
        print(f"      Вероятность эпидемии: {avg_metrics['epidemic_probability']*100:.1f}%")
        print(f"      Угроз: {avg_metrics['avg_threat_messages']:.1f} ± {avg_metrics['std_threat_messages']:.1f}")
    
    # ============================================================
    # СОЗДАНИЕ ГРАФИКОВ
    # ============================================================
    print("\n" + "=" * 60)
    print("📈 СОЗДАНИЕ ГРАФИКОВ")
    print("=" * 60)
    
    # Подготовка данных для графиков
    red_counts = [r["red_count"] for r in all_results]
    infection_ratios = [r["avg_infection_ratio"] for r in all_results]
    infection_stds = [r["std_infection_ratio"] for r in all_results]
    threat_counts = [r["avg_threat_messages"] for r in all_results]
    threat_stds = [r["std_threat_messages"] for r in all_results]
    epidemic_probs = [r["epidemic_probability"] for r in all_results]
    
    # График 1: Заражение с ошибками
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Анализ распространения угроз в зависимости от количества красных узлов', 
                 fontsize=16, fontweight='bold')
    
    # График доли заражённых
    ax1 = axes[0, 0]
    ax1.errorbar(red_counts, infection_ratios, yerr=infection_stds, 
                 fmt='ro-', capsize=5, capthick=2, elinewidth=2, 
                 markersize=8, linewidth=2, label='Доля заражённых')
    ax1.axhline(y=INFECTED_NODES_RATIO, color='r', linestyle='--', 
                alpha=0.7, label=f'Порог эпидемии ({INFECTED_NODES_RATIO*100}%)')
    ax1.set_xlabel('Количество красных узлов', fontsize=12)
    ax1.set_ylabel('Доля заражённых пользователей', fontsize=12)
    ax1.set_title('Заражение системы', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График количества сообщений
    ax2 = axes[0, 1]
    ax2.errorbar(red_counts, threat_counts, yerr=threat_stds, 
                 fmt='r^-', capsize=5, capthick=2, elinewidth=2,
                 markersize=8, linewidth=2, label='Угрозы')
    ax2.set_xlabel('Количество красных узлов', fontsize=12)
    ax2.set_ylabel('Количество сообщений', fontsize=12)
    ax2.set_title('Активность в системе', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # График вероятности эпидемии
    ax3 = axes[1, 0]
    ax3.plot(red_counts, epidemic_probs, 'gs-', linewidth=2, markersize=8, label='Вероятность эпидемии')
    ax3.fill_between(red_counts, 0, epidemic_probs, alpha=0.3, color='green')
    ax3.set_xlabel('Количество красных узлов', fontsize=12)
    ax3.set_ylabel('Вероятность эпидемии', fontsize=12)
    ax3.set_title('Риск эпидемии', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1.05])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Корреляционный график
    ax4 = axes[1, 1]
    ax4.scatter(red_counts, infection_ratios, s=100, alpha=0.7, 
                c=red_counts, cmap='viridis', edgecolors='black')
    # Полиномиальная аппроксимация
    z = np.polyfit(red_counts, infection_ratios, 2)
    p = np.poly1d(z)
    ax4.plot(red_counts, p(red_counts), "b--", alpha=0.8, 
             label=f'Аппроксимация (R²={np.corrcoef(red_counts, infection_ratios)[0,1]**2:.3f})')
    ax4.set_xlabel('Количество красных узлов', fontsize=12)
    ax4.set_ylabel('Доля заражённых', fontsize=12)
    ax4.set_title('Корреляционный анализ', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'infection_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # ============================================================
    # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ============================================================
    print("\n" + "=" * 60)
    print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    # Создаём сводную таблицу
    summary = []
    for r in all_results:
        summary.append({
            "red_nodes": r["red_count"],
            "avg_infected": r["avg_infected_users"],
            "std_infected": r["std_infected_users"],
            "infection_ratio": r["avg_infection_ratio"],
            "std_infection_ratio": r["std_infection_ratio"],
            "epidemic_probability": r["epidemic_probability"],
            "avg_threats": r["avg_threat_messages"],
            "avg_warnings": r["avg_warning_messages"]
        })
    
    # Сохраняем в JSON
    with open(os.path.join(experiment_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Сохраняем в CSV
    import csv
    with open(os.path.join(experiment_dir, "summary.csv"), "w", newline='', encoding="utf-8") as f:
        if summary:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
    
    # Сохраняем мета-информацию
    meta = {
        "timestamp": timestamp,
        "config": {
            "n_users": N_USERS,
            "n_red_values": N_RED_VALUES,
            "n_llm": N_LLM,
            "n_blue": N_BLUE,
            "avg_degree": AVG_DEGREE,
            "t_steps": T_STEPS,
            "n_runs": N_RUNS,
            "infection_threshold": INFECTION_THRESHOLD,
            "epidemic_ratio": INFECTED_NODES_RATIO
        },
        "messages_path": MESSAGES_PATH
    }
    
    with open(os.path.join(experiment_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    # Определяем критический порог
    print("\n" + "=" * 60)
    print("🎯 ОПРЕДЕЛЕНИЕ ПОРОГА ЭПИДЕМИИ")
    print("=" * 60)
    
    epidemic_threshold = None
    for r in all_results:
        if r["epidemic_probability"] >= 0.5:
            epidemic_threshold = r["red_count"]
            break
    
    if epidemic_threshold:
        print(f"⚠️ Критический порог (50% вероятность эпидемии): {epidemic_threshold} красных узлов")
        
        # Находим безопасный максимум
        safe_max = None
        for r in all_results:
            if r["red_count"] < epidemic_threshold and r["epidemic_probability"] < 0.1:
                safe_max = r["red_count"]
        
        if safe_max:
            print(f"   Безопасный максимум (вероятность <10%): {safe_max} красных узлов")
            print(f"   Окно уязвимости: {safe_max + 1} - {epidemic_threshold}")
    else:
        print("⚠️ В исследуемом диапазоне не достигнут порог 50% вероятности эпидемии")
    
    print("\n" + "=" * 60)
    print("✅ ЭКСПЕРИМЕНТ ЗАВЕРШЁН УСПЕШНО!")
    print("=" * 60)
    print(f"📁 Результаты сохранены в: {experiment_dir}")
    print(f"📊 Сводная таблица: {os.path.join(experiment_dir, 'summary.csv')}")
    print(f"📈 Графики: {os.path.join(experiment_dir, 'infection_analysis.png')}")
    
    return all_results

# ============================================================
# ЗАПУСК
# ============================================================
if __name__ == "__main__":
    try:
        results = run_experiment()
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении эксперимента: {e}")
        import traceback
        traceback.print_exc()
        input("\nНажмите Enter для выхода...")