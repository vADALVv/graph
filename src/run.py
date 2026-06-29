# run.py
from graph_structure import create_graph
from simulation import simulate_diffusion
from visualization import visualize_graph
from blue_agent import BlueAgent
from collections import defaultdict
import json
import os

N_USERS = 120
N_RED = 4
N_LLM = 3
N_BLUE = 10
AVG_DEGREE = 4
T_STEPS = 40

MESSAGES_PATH = r"C:\Users\Vlada\Desktop\llm_attaks\graph\data\messages.json"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "simulation_result.json")
VIZ_PATH = os.path.join(OUTPUT_DIR, "network_visualization_pro.html")

# ------------------------------------------------------------
# НАСТРОЙКА РЕЖИМА LLM
# ------------------------------------------------------------
USE_REAL_LLM = False   # Поставьте True, если хотите использовать реальную модель TinyLlama

print("=" * 50)
print("🔧 CREATING GRAPH")
print("=" * 50)

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

# ------------------------------------------------------------
blue_agent = None
if N_BLUE > 0:
    print("\n" + "=" * 50)
    print("🔷 INITIALIZING BLUE AGENT (Moderator)")
    print("=" * 50)
    blue_agent = BlueAgent()
    print("✅ Blue Agent ready for real-time moderation")

# ------------------------------------------------------------
print("\n" + "=" * 50)
print("🔄 RUNNING SIMULATION")
print("=" * 50)

results = simulate_diffusion(
    G=G,
    users=users,
    node_types=node_types,
    T_steps=T_STEPS,
    messages_path=MESSAGES_PATH,
    blue_agent=blue_agent,
    use_real_llm=USE_REAL_LLM      # Передаём настройку
)

timeline = results["timeline"]
states_history = results.get("states_history", [])
risk_scores = results.get("risk_scores", {})
risk_levels = results.get("risk_levels", {})
blocked_messages = results.get("blocked_messages", 0)

print(f"\n✅ Simulation complete:")
print(f"   - Timeline events: {len(timeline)}")
print(f"   - Blocked messages: {blocked_messages}")

# ------------------------------------------------------------
print("\n" + "=" * 50)
print("📊 STATISTICS")
print("=" * 50)

threat_count = sum(1 for e in timeline if e.get('category') == 'threat')
manipulative_count = sum(1 for e in timeline if e.get('category') == 'manipulative')
neutral_count = sum(1 for e in timeline if e.get('category') == 'neutral')
warning_count = sum(1 for e in timeline if e.get('category') == 'warning')

print(f"   - Threat: {threat_count}")
print(f"   - Manipulative: {manipulative_count}")
print(f"   - Neutral: {neutral_count}")
print(f"   - Warnings: {warning_count}")
print(f"   - Blocked: {blocked_messages}")

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
    "blocked_messages": blocked_messages,
    "simulation_params": {
        "n_users": N_USERS, "n_red": N_RED, "n_llm": N_LLM, "n_blue": N_BLUE,
        "avg_degree": AVG_DEGREE, "t_steps": T_STEPS,
        "total_events": len(timeline)
    }
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✅ Saved to: {OUTPUT_PATH}")

# ------------------------------------------------------------
print("\n" + "=" * 50)
print("🎨 GENERATING VISUALIZATION")
print("=" * 50)

try:
    visualize_graph(
        G=G, results=results, users=users,
        node_types=node_types, blue_agent=blue_agent,
        output_path=VIZ_PATH
    )
    print(f"✅ Visualization: {VIZ_PATH}")
except Exception as e:
    print(f"❌ Visualization error: {e}")

print("\n" + "=" * 50)
print("✅ ALL DONE!")
print("=" * 50)
print(f"\n💡 Open: {VIZ_PATH}")