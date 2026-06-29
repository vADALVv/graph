import math
import random
import copy
from dataclasses import dataclass
from typing import Dict, Optional, List, Set, Tuple
from collections import defaultdict
import networkx as nx

@dataclass
class Message:
    msg_id: int
    text: str
    b: float
    h: float
    src: int
    t: int
    category: str

@dataclass
class UserState:
    b: float
    c: float
    e: float

@dataclass
class RepostParams:
    lambda0: float = -1.0
    lambda1: float = 2.0
    lambda2: float = 1.8
    lambda3: float = 3.0
    lambda4: float = 1.2
    alpha: float = 1.0
    beta: float = 0.2

P_USER = 0.9
P_RED = 0.7
P_LLM = 0.6
P_BLUE = 0.0

BLUE_DETECTION_THRESHOLD = 0.6

neutral_messages = []
threat_messages = []
manipulative_messages = []
_msg_counter = 0

def init_message_bank(path: str):
    import json
    global neutral_messages, threat_messages, manipulative_messages
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    neutral_messages.clear()
    threat_messages.clear()
    manipulative_messages.clear()
    for item in data:
        msg = {"text": item["text"], "b": float(item["b"]), "h": float(item["h"])}
        t = item.get("type", "neutral")
        if t == "neutral":
            neutral_messages.append(msg)
        elif t == "threat":
            threat_messages.append(msg)
        else:
            manipulative_messages.append(msg)

def sigmoid(x: float):
    return 1 / (1 + math.exp(-max(-10, min(10, x))))

def kappa(b_i, b_m, alpha):
    return math.exp(-alpha * abs(b_m - b_i))

def update_user(state: UserState, msg: Message, rp: RepostParams):
    k = kappa(state.b, msg.b, rp.alpha)
    e_new = state.e * 0.97 + msg.h * 0.35
    e_new = max(-3, min(3, e_new))
    b_new = state.b + state.c * rp.beta * k * (msg.b - state.b)
    b_new = max(-1, min(1, b_new))
    c_new = state.c + 0.003 * (k - 0.5)
    c_new = max(0.05, min(1.0, c_new))
    return UserState(b_new, c_new, e_new), k

def get_transmission_probability(node_type: str) -> float:
    probs = {"U": P_USER, "R": P_RED, "L": P_LLM, "B": P_BLUE}
    return probs.get(node_type, P_USER)

def blue_agent_detect(message: Message, blue_agent):
    if blue_agent:
        return blue_agent.process_event(message.text)
    text_lower = message.text.lower()
    if any(w in text_lower for w in ["danger", "warning", "alert", "urgent", "crisis"]):
        return 0.7, "threat"
    elif any(w in text_lower for w in ["trust", "believe", "must", "never"]):
        return 0.5, "manipulative"
    return 0.2, "neutral"

def generate_warning(original_msg: Message, msg_id: int, t: int):
    return Message(
        msg_id=msg_id,
        text=f"⚠️ BLUE MODERATOR ALERT: Potentially harmful content detected! (Source: {original_msg.src})",
        b=0.0, h=0.1, src=-1, t=t, category="warning"
    )

def generate(node_type: str, node_id: int, t: int):
    global _msg_counter
    if node_type == "U":
        pool = neutral_messages
    elif node_type == "R":
        pool = threat_messages + manipulative_messages
    elif node_type == "B":
        return None
    else:
        pool = neutral_messages + threat_messages + manipulative_messages
    if not pool:
        return None
    m = random.choice(pool)
    if m in threat_messages:
        cat = "threat"
    elif m in manipulative_messages:
        cat = "manipulative"
    else:
        cat = "neutral"
    msg = Message(
        msg_id=_msg_counter,
        text=m["text"],
        b=m["b"],
        h=m["h"],
        src=node_id,
        t=t,
        category=cat
    )
    _msg_counter += 1
    return msg

def calculate_risk_scores(timeline, state, node_types):
    risk_scores = {}
    risk_levels = {}
    received_h = defaultdict(list)
    for event in timeline:
        target = event.get("to")
        if target is None or target == -1:
            continue
        category = event.get("category", "")
        if category.startswith("detected"):
            continue
        h = event.get("h", 0)
        if category == "warning":
            h *= 0.1
        received_h[target].append(h)
    for node in state.keys():
        if node_types.get(node) in ("B", "L"):
            risk_scores[node] = 0.0
            risk_levels[node] = "UNKNOWN"
            continue
        if received_h[node]:
            score = max(received_h[node])
        else:
            score = 0.0
        risk_scores[node] = score
        if score < 0.4:
            level = "LOW"
        elif score < 0.6:
            level = "MEDIUM"
        else:
            level = "HIGH"
        risk_levels[node] = level
    return risk_scores, risk_levels

def simulate_diffusion(
    G: nx.DiGraph,
    users: Dict[int, UserState],
    node_types: Dict[int, str],
    T_steps: int = 10,
    rp: RepostParams = RepostParams(),
    seed: int = 42,
    messages_path: Optional[str] = None,
    blue_agent = None,
    use_real_llm: bool = False   # НОВЫЙ ПАРАМЕТР
):
    random.seed(seed)
    if messages_path:
        init_message_bank(messages_path)

    state = copy.deepcopy(users)
    timeline = []
    history = []
    global _msg_counter
    _msg_counter = 0

    infected: Dict[int, Set[int]] = defaultdict(set)
    frontier: List[Tuple[int, Message, int]] = []
    edge_busy: Dict[Tuple[int, int], Tuple[int, int]] = {}

    # ---------- LLM МЕНЕДЖЕР ----------
    llm_manager = None
    if 'L' in node_types.values():
        from llm_agents import LLMAgentManager
        fallback_mode = not use_real_llm
        print(f"\n🤖 LLM AGENTS DETECTED: using {'REAL LLM (TinyLlama)' if not fallback_mode else 'FALLBACK (mock messages)'}")
        llm_manager = LLMAgentManager(use_fallback=fallback_mode, toxicity_probability=0.3)
        for node_id, ntype in node_types.items():
            if ntype == "L":
                llm_manager.init_agent(node_id, persona="social media user", belief=0.0)

    history.append({k: v.__dict__ for k, v in state.items()})

    # ---------- НАЧАЛЬНАЯ ГЕНЕРАЦИЯ (t=0) ----------
    for node_id, ntype in node_types.items():
        if ntype == "B":
            continue
        gen_prob = P_USER if ntype == "U" else (P_RED if ntype == "R" else P_LLM)
        if random.random() < gen_prob:
            if ntype == "L" and llm_manager:
                msg_data = llm_manager.generate_message(node_id, 0)
                if not msg_data:
                    continue
                msg = Message(
                    msg_id=_msg_counter,
                    text=msg_data["message"],
                    b=0.0,
                    h=msg_data["h"],
                    src=node_id,
                    t=0,
                    category=msg_data["category"]
                )
                _msg_counter += 1
            else:
                msg = generate(ntype, node_id, 0)
            if msg:
                frontier.append((node_id, msg, 0))
                infected[msg.msg_id].add(node_id)
                timeline.append({
                    "t": 0,
                    "from": node_id,
                    "to": None,
                    "msg_id": msg.msg_id,
                    "text": msg.text,
                    "category": msg.category,
                    "b": msg.b,
                    "h": msg.h,
                    "age": 0
                })

    # ---------- ОСНОВНОЙ ЦИКЛ ----------
    for t in range(0, T_steps):
        new_frontier = []
        reposted_this_step = set()
        used_edges_this_step = set()

        for edge, (_, release_step) in list(edge_busy.items()):
            if release_step <= t:
                del edge_busy[edge]

        for sender, msg, age in frontier:
            sender_type = node_types.get(sender, "U")
            transmission_prob = 1.0 if sender_type == "B" else get_transmission_probability(sender_type)
            is_warning = (msg.category == "warning")

            for receiver in G.successors(sender):
                edge = (sender, receiver)
                if edge in edge_busy or edge in used_edges_this_step:
                    continue
                if receiver in reposted_this_step or receiver in infected[msg.msg_id]:
                    continue
                if random.random() > transmission_prob:
                    continue

                # ----- СИНИЙ МОДЕРАТОР -----
                if node_types.get(receiver) == "B" and not is_warning:
                    detected_risk, detected_category = blue_agent_detect(msg, blue_agent)
                    timeline.append({
                        "t": t,
                        "from": sender,
                        "to": receiver,
                        "msg_id": msg.msg_id,
                        "text": msg.text,
                        "category": f"detected_{detected_category}",
                        "b": msg.b,
                        "h": msg.h,
                        "age": age,
                        "detected_risk": detected_risk
                    })
                    if detected_risk > BLUE_DETECTION_THRESHOLD:
                        warning_msg = generate_warning(msg, _msg_counter, t)
                        _msg_counter += 1
                        new_frontier.append((receiver, warning_msg, 0))
                        reposted_this_step.add(receiver)
                        used_edges_this_step.add(edge)
                        edge_busy[edge] = (msg.msg_id, t + 1)
                        continue

                # ----- РЕПОСТ (только U и R) -----
                if node_types.get(receiver) not in ("U", "R"):
                    continue

                rel = G[sender][receiver].get("weight", 1.0)
                k_val = kappa(state[receiver].b, msg.b, rp.alpha)
                p = sigmoid(
                    rp.lambda0 + rp.lambda1 * k_val +
                    rp.lambda2 * state[receiver].e +
                    rp.lambda3 * msg.h + rp.lambda4 * rel
                ) * (0.99 ** age)

                if random.random() < p:
                    if node_types.get(receiver) == "U":
                        state[receiver], _ = update_user(state[receiver], msg, rp)
                    infected[msg.msg_id].add(receiver)
                    reposted_this_step.add(receiver)
                    used_edges_this_step.add(edge)
                    new_frontier.append((receiver, msg, age + 1))
                    timeline.append({
                        "t": t + 1,
                        "from": sender,
                        "to": receiver,
                        "msg_id": msg.msg_id,
                        "text": msg.text,
                        "category": msg.category,
                        "b": msg.b,
                        "h": msg.h,
                        "age": age + 1,
                        "state_b": state[receiver].b,
                        "state_c": state[receiver].c,
                        "state_e": state[receiver].e
                    })
                    edge_busy[edge] = (msg.msg_id, t + 1)

        # ----- ГЕНЕРАЦИЯ НОВЫХ СООБЩЕНИЙ -----
        for node_id, ntype in node_types.items():
            if ntype == "B":
                continue
            if node_id in reposted_this_step:
                continue
            gen_prob = P_USER if ntype == "U" else (P_RED if ntype == "R" else P_LLM)
            if random.random() < gen_prob:
                if ntype == "L" and llm_manager:
                    msg_data = llm_manager.generate_message(node_id, t)
                    if not msg_data:
                        continue
                    msg = Message(
                        msg_id=_msg_counter,
                        text=msg_data["message"],
                        b=0.0,
                        h=msg_data["h"],
                        src=node_id,
                        t=t,
                        category=msg_data["category"]
                    )
                    _msg_counter += 1
                else:
                    msg = generate(ntype, node_id, t)
                if msg:
                    new_frontier.append((node_id, msg, 0))
                    infected[msg.msg_id].add(node_id)
                    timeline.append({
                        "t": t,
                        "from": node_id,
                        "to": None,
                        "msg_id": msg.msg_id,
                        "text": msg.text,
                        "category": msg.category,
                        "b": msg.b,
                        "h": msg.h,
                        "age": 0
                    })

        frontier = new_frontier
        history.append({k: v.__dict__ for k, v in state.items()})

    risk_scores, risk_levels = calculate_risk_scores(timeline, state, node_types)
    formatted_history = [{str(nid): sd for nid, sd in snap.items()} for snap in history]

    return {
        "users_final": {k: v.__dict__ for k, v in state.items()},
        "timeline": timeline,
        "states_history": formatted_history,
        "total_messages": _msg_counter,
        "risk_scores": risk_scores,
        "risk_levels": risk_levels
    }