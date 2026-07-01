# run.py
from graph_structure import create_graph
from simulation import simulate_diffusion, RepostParams
from visualization import visualize_graph
from blue_agent import BlueAgent
from collections import defaultdict
import json
import os

# ============================================================
# КЛЮЧИ ДЛЯ API — все берутся из config.py (заполнять там)
# ============================================================
import config
from config import (
    GIGACHAT_AUTH_KEY, GIGACHAT_SCOPE,
    YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEX_IAM_TOKEN,
    HF_TOKEN,
    LLM_MODEL_TYPE, BLUE_MODEL_TYPE,
)

# ============================================================
# НАСТРОЙКИ СИМУЛЯЦИИ
# ============================================================
N_USERS = 30
N_RED = 3
N_LLM = 2
N_BLUE = 4
AVG_DEGREE = 4
T_STEPS = 10

# ============================================================
# ЗАЩИТА И КОНТРФАКТУАЛ
# ============================================================
# DEFENSE_POLICY = None  -> модератор только наблюдает (порога/блокировки нет).
# DEFENSE_POLICY = 0.6   -> источник с оценкой риска >= 0.6 уходит в карантин.
DEFENSE_POLICY = None
RUN_COUNTERFACTUAL = False   # True -> прогнать без защиты и с защитой, сравнить метрики
COUNTERFACTUAL_POLICY = 0.6
BELIEF_THRESHOLD = 0.5       # порог убеждения для метрик радикализации

# ============================================================
# НАСТРОЙКИ ЛОКАЛЬНЫХ МОДЕЛЕЙ
# ============================================================
MODEL_DIR = None
LOAD_IN_4BIT = True
DEVICE = "auto"
MAX_LENGTH = 2048
BATCH_SIZE = 32

# ============================================================

MESSAGES_PATH = r"C:\Users\Vlada\Desktop\llm_attaks\graph\data\messages.json"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "simulation_result.json")
VIZ_PATH = os.path.join(OUTPUT_DIR, "network_visualization_pro.html")

print("=" * 50)
print("🔧 CREATING GRAPH")
print("=" * 50)

# Предупреждения о типичных ошибках заполнения ключей
config.check_config(LLM_MODEL_TYPE)
config.check_config(BLUE_MODEL_TYPE)

if not os.path.exists(MESSAGES_PATH):
    print(f"❌ Error: Messages file not found at {MESSAGES_PATH}")
    exit(1)

G, users, node_types = create_graph(
    num_u=N_USERS,
    num_r=N_RED,
    num_l=N_LLM,
    num_b=N_BLUE,
    avg_degree=AVG_DEGREE
)

print(f"✅ Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"   - LLM agents in graph: {N_LLM}")
print(f"   - Blue agents in graph: {N_BLUE}")

# ------------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ BLUE AGENT
# ------------------------------------------------------------
blue_agent = None
if N_BLUE > 0:
    print("\n" + "=" * 50)
    print("🔷 INITIALIZING BLUE AGENT (Moderator)")
    print("=" * 50)
    
    try:
        blue_agent = BlueAgent(
            model_type=BLUE_MODEL_TYPE,
            model_dir=MODEL_DIR,
            max_length=MAX_LENGTH,
            device=DEVICE,
            load_in_4bit=LOAD_IN_4BIT,
            batch_size=BATCH_SIZE,
            gigachat_auth_key=GIGACHAT_AUTH_KEY,
            gigachat_scope=GIGACHAT_SCOPE,
            yandex_api_key=YANDEX_API_KEY,
            yandex_folder_id=YANDEX_FOLDER_ID,
            yandex_iam_token=YANDEX_IAM_TOKEN,
            hf_token=HF_TOKEN
        )
        
        print("\n   Testing Blue Agent...")
        test_messages = [
            "Привет! Как дела?",
            "Это опасно! Нужно срочно действовать!"
        ]
        
        for msg in test_messages:
            result = blue_agent.analyze(msg)
            rs = result["RiskScore"]
            rs_str = f"{rs:.3f}" if rs is not None else "FAILED"
            print(f"      - '{msg[:30]}...' → Risk: {rs_str} ({result['Level']})")
        
        print(f"\n✅ Blue Agent ready")
        print(f"   - Model: {blue_agent.model_type}")
        print(f"   - Device: {blue_agent.device}")
        print(f"   - Is API: {blue_agent.is_api_model}")
            
    except Exception as e:
        print(f"❌ Failed to initialize Blue Agent: {e}")
        print("   Simulation will continue WITHOUT moderation")
        blue_agent = None

# ------------------------------------------------------------
print("\n" + "=" * 50)
print("🔄 RUNNING SIMULATION")
print("=" * 50)
print(f"📌 LLM Model: {LLM_MODEL_TYPE.upper()}")
print(f"📌 Blue Model: {BLUE_MODEL_TYPE.upper() if blue_agent else 'DISABLED'}")
print(f"📌 Steps: {T_STEPS}")
print(f"📌 Users: {N_USERS} (R:{N_RED}, L:{N_LLM}, B:{N_BLUE})")

try:
    results = simulate_diffusion(
        G=G,
        users=users,
        node_types=node_types,
        T_steps=T_STEPS,
        rp=RepostParams(),
        seed=42,
        messages_path=MESSAGES_PATH,
        blue_agent=blue_agent,
        llm_model_type=LLM_MODEL_TYPE,
        gigachat_auth_key=GIGACHAT_AUTH_KEY,
        gigachat_scope=GIGACHAT_SCOPE,
        yandex_api_key=YANDEX_API_KEY,
        yandex_folder_id=YANDEX_FOLDER_ID,
        yandex_iam_token=YANDEX_IAM_TOKEN,
        hf_token=HF_TOKEN,
        defense_policy=DEFENSE_POLICY
    )
except Exception as e:
    print(f"❌ Simulation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

timeline = results["timeline"]
states_history = results.get("states_history", [])
risk_scores = results.get("risk_scores", {})
risk_levels = results.get("risk_levels", {})

from metrics import compute_diffusion_metrics, compute_blue_stats, format_metrics

blue_stats = compute_blue_stats(results)
quarantined = results.get("quarantined", [])

print(f"\n✅ Simulation complete:")
print(f"   - Timeline events: {len(timeline)}")
print(f"   - Unique messages: {results.get('total_messages', 0)}")
print(f"   - Blue assessments: {blue_stats['assessed_valid']} valid, {blue_stats['assessed_failed']} failed")
print(f"   - Quarantined sources: {len(quarantined)} {quarantined if quarantined else ''}")

# ------------------------------------------------------------
print("\n" + "=" * 50)
print("📊 STATISTICS")
print("=" * 50)

threat_count = sum(1 for e in timeline if e.get('category') == 'threat')
manipulative_count = sum(1 for e in timeline if e.get('category') == 'manipulative')
neutral_count = sum(1 for e in timeline if e.get('category') == 'neutral')
warning_count = sum(1 for e in timeline if e.get('category') == 'warning')
detected_count = sum(1 for e in timeline if e.get('category', '').startswith('detected_'))
toxic_count = sum(1 for e in timeline if e.get('category') == 'toxic')

print(f"   - Threat messages: {threat_count}")
print(f"   - Manipulative messages: {manipulative_count}")
print(f"   - Neutral messages: {neutral_count}")
print(f"   - Toxic messages: {toxic_count}")
print(f"   - Warnings (from Blue): {warning_count}")
print(f"   - Detected (assessed by Blue): {detected_count}")
print(f"   - Blue failed assessments: {blue_stats['assessed_failed']}")
print(f"   - Quarantine actions: {blue_stats['quarantine_actions']}")

# ------------------------------------------------------------
print("\n" + "=" * 50)
print("📈 DIFFUSION METRICS")
print("=" * 50)
diffusion_metrics = compute_diffusion_metrics(results, node_types, belief_threshold=BELIEF_THRESHOLD)
print(format_metrics(diffusion_metrics))

print("\n   Activity by node type:")
type_activity = defaultdict(int)
for e in timeline:
    if e.get("from") is not None:
        from_type = node_types.get(e["from"], "U")
        type_activity[from_type] += 1

type_names = {0: "U (User)", 1: "R (Red)", 2: "L (LLM)", 3: "B (Blue)"}
for ntype, count in sorted(type_activity.items()):
    type_name = type_names.get(ntype, str(ntype))
    percentage = (count / len(timeline) * 100) if timeline else 0
    print(f"      - {type_name}: {count} events ({percentage:.1f}%)")

if risk_scores:
    avg_risk = sum(risk_scores.values()) / len(risk_scores)
    high_risk = sum(1 for v in risk_scores.values() if v >= 0.6)
    med_risk = sum(1 for v in risk_scores.values() if 0.4 <= v < 0.6)
    low_risk = sum(1 for v in risk_scores.values() if v < 0.4)
    print(f"\n   Risk scores:")
    print(f"      - Average: {avg_risk:.3f}")
    print(f"      - High risk (>=0.6): {high_risk}")
    print(f"      - Medium risk (0.4-0.6): {med_risk}")
    print(f"      - Low risk (<0.4): {low_risk}")

# ------------------------------------------------------------
edge_counts = defaultdict(int)
edge_risk = defaultdict(float)

for e in timeline:
    if e.get("to") is not None and e.get("to") != -1:
        k = (e["from"], e["to"])
        edge_counts[k] += 1
        edge_risk[k] += e.get("h", 0)

edge_metrics = {
    f"{u}->{v}": {"reposts": edge_counts[(u, v)], "risk_sum": edge_risk[(u, v)]}
    for (u, v) in edge_counts
}

if edge_metrics:
    top_edges = sorted(edge_metrics.items(), key=lambda x: x[1]["reposts"], reverse=True)[:5]
    print(f"\n   Top 5 active edges:")
    for edge, metrics in top_edges:
        print(f"      - {edge}: {metrics['reposts']} reposts, risk: {metrics['risk_sum']:.3f}")

# ------------------------------------------------------------
print("\n" + "=" * 50)
print("💾 SAVING RESULTS")
print("=" * 50)

nodes_final = {}
for k, v in results["users_final"].items():
    if hasattr(v, 'b'):
        nodes_final[str(k)] = {"b": v.b, "c": v.c, "e": v.e}
    else:
        nodes_final[str(k)] = v

output = {
    "nodes": nodes_final,
    "node_types": {str(k): v for k, v in node_types.items()},
    "states_history": states_history,
    "edges": edge_metrics,
    "timeline": timeline,
    "risk_scores": {str(k): v for k, v in risk_scores.items()},
    "risk_levels": {str(k): v for k, v in risk_levels.items()},
    "diffusion_metrics": diffusion_metrics,
    "blue_stats": blue_stats,
    "quarantined": quarantined,
    "simulation_params": {
        "n_users": N_USERS, 
        "n_red": N_RED, 
        "n_llm": N_LLM, 
        "n_blue": N_BLUE,
        "avg_degree": AVG_DEGREE, 
        "t_steps": T_STEPS,
        "total_events": len(timeline),
        "llm_model_type": LLM_MODEL_TYPE,
        "blue_model_type": BLUE_MODEL_TYPE if blue_agent else "DISABLED",
        "defense_policy": DEFENSE_POLICY,
        "belief_threshold": BELIEF_THRESHOLD,
        "device": DEVICE
    }
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✅ Saved to: {OUTPUT_PATH}")
print(f"   File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")

# ------------------------------------------------------------
# КОНТРФАКТУАЛ "С ЗАЩИТОЙ / БЕЗ"
# ------------------------------------------------------------
if RUN_COUNTERFACTUAL:
    print("\n" + "=" * 50)
    print("🧪 COUNTERFACTUAL: no defense vs defense")
    print("=" * 50)
    from metrics import run_counterfactual

    def make_world():
        return create_graph(
            num_u=N_USERS, num_r=N_RED, num_l=N_LLM,
            num_b=N_BLUE, avg_degree=AVG_DEGREE
        )

    sim_kwargs = dict(
        T_steps=T_STEPS,
        rp=RepostParams(),
        seed=42,
        messages_path=MESSAGES_PATH,
        blue_agent=blue_agent,
        llm_model_type=LLM_MODEL_TYPE,
        gigachat_auth_key=GIGACHAT_AUTH_KEY,
        gigachat_scope=GIGACHAT_SCOPE,
        yandex_api_key=YANDEX_API_KEY,
        yandex_folder_id=YANDEX_FOLDER_ID,
        yandex_iam_token=YANDEX_IAM_TOKEN,
        hf_token=HF_TOKEN,
    )

    cf = run_counterfactual(
        make_world=make_world,
        sim_fn=simulate_diffusion,
        sim_kwargs=sim_kwargs,
        belief_threshold=BELIEF_THRESHOLD,
        defense_policy=COUNTERFACTUAL_POLICY,
    )

    print("\n   БЕЗ защиты:")
    print(format_metrics(cf["no_defense"]))
    print("\n   С защитой:")
    print(format_metrics(cf["with_defense"]))
    print("\n   Δ (защита − без защиты):")
    for key in ("cascade_size_mean", "cascade_depth_mean", "r0_per_seed",
                "frac_radicalized", "polarization_var", "n_spreads"):
        print(f"      - {key}: {cf['delta'].get(key, 0):+.4f}")

    cf_out = os.path.join(OUTPUT_DIR, "counterfactual.json")
    with open(cf_out, "w", encoding="utf-8") as f:
        json.dump(
            {"no_defense": cf["no_defense"], "with_defense": cf["with_defense"], "delta": cf["delta"]},
            f, ensure_ascii=False, indent=2
        )
    print(f"\n✅ Counterfactual saved: {cf_out}")

# ------------------------------------------------------------
print("\n" + "=" * 50)
print("🎨 GENERATING VISUALIZATION")
print("=" * 50)

try:
    visualize_graph(
        G=G, 
        results=results, 
        users=users,
        node_types=node_types, 
        blue_agent=blue_agent,
        output_path=VIZ_PATH
    )
    print(f"✅ Visualization saved: {VIZ_PATH}")
except Exception as e:
    print(f"❌ Visualization error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("✅ ALL DONE!")
print("=" * 50)
print(f"\n💡 Open visualization: {VIZ_PATH}")
print(f"💡 Results JSON: {OUTPUT_PATH}")

if blue_agent:
    stats = blue_agent.get_stats()
    print(f"\n📊 Blue Agent Stats:")
    print(f"   - Total analyzed: {stats.get('total_analyzed', 0)}")
    print(f"   - Cache size: {stats.get('cache_size', 0)}")
    print(f"   - Average risk: {stats.get('avg_risk', 0):.3f}")