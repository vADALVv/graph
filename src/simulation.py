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
    # --- эмоциональная динамика ---
    h_baseline: float = 0.5   # нейтральный уровень возбуждения сообщения
    e_relax: float = 0.9      # релаксация эмоции к 0 (затухание возбуждения)
    e_gain: float = 0.35      # чувствительность к импульсу (h - h_baseline)

P_USER = 0.9
P_RED = 0.7
P_LLM = 0.6
P_BLUE = 0.0

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
    # Знаковый эмоциональный импульс: h < baseline успокаивает, h > baseline возбуждает,
    # плюс релаксация к нулю — устойчивая точка равновесия вместо одностороннего дрейфа вверх.
    e_new = rp.e_relax * state.e + rp.e_gain * (msg.h - rp.h_baseline)
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
    llm_model_type: str = "t-lite",
    gigachat_auth_key: Optional[str] = None,
    gigachat_scope: str = "GIGACHAT_API_PERS",
    yandex_api_key: Optional[str] = None,
    yandex_folder_id: Optional[str] = None,
    yandex_iam_token: Optional[str] = None,
    hf_token: Optional[str] = None,
    defense_policy: Optional[float] = None,
    verbose: bool = True
):
    """
    defense_policy:
      None  — модератор только наблюдает и оценивает (порога/блокировки нет).
      float — активная защита: при оценке риска >= defense_policy источник
              сообщения отправляется в карантин (перестаёт распространять).
              Используется для контрфактуала "с защитой / без".
    """
    random.seed(seed)
    if messages_path:
        init_message_bank(messages_path)

    # ---- ОТЛАДКА: проверка банка ----
    if verbose:
        print(f"📚 Bank sizes: neutral={len(neutral_messages)}, threat={len(threat_messages)}, manip={len(manipulative_messages)}")

    state = copy.deepcopy(users)
    timeline = []
    history = []
    global _msg_counter
    _msg_counter = 0

    infected: Dict[int, Set[int]] = defaultdict(set)
    frontier: List[Tuple[int, Message, int]] = []
    edge_busy: Dict[Tuple[int, int], Tuple[int, int]] = {}
    quarantined: Set[int] = set()   # источники, заблокированные активной защитой

    # ---------- LLM МЕНЕДЖЕР ----------
    llm_manager = None
    if 'L' in node_types.values():
        from llm_agent import LLMAgentManager
        if verbose:
            print(f"\n🤖 LLM AGENTS DETECTED: using {llm_model_type.upper()}")
        
        if llm_model_type == "t-lite":
            llm_manager = LLMAgentManager(
                toxicity_probability=0.3,
                model_type="t-lite"
            )
        elif llm_model_type == "gigachat":
            if not gigachat_auth_key:
                raise ValueError("GigaChat auth key is required for model_type='gigachat'")
            llm_manager = LLMAgentManager(
                toxicity_probability=0.3,
                model_type="gigachat",
                gigachat_auth_key=gigachat_auth_key,
                gigachat_scope=gigachat_scope
            )
        elif llm_model_type == "yandexgpt":
            if not yandex_folder_id:
                raise ValueError("Yandex GPT folder_id is required")
            if not yandex_api_key and not yandex_iam_token:
                raise ValueError("Yandex GPT requires either api_key or iam_token")
            llm_manager = LLMAgentManager(
                toxicity_probability=0.3,
                model_type="yandexgpt",
                yandex_api_key=yandex_api_key,
                yandex_folder_id=yandex_folder_id,
                yandex_iam_token=yandex_iam_token
            )
        elif llm_model_type == "deepseek":
            print("⚠️ DeepSeek support coming soon, falling back to t-lite")
            llm_manager = LLMAgentManager(
                toxicity_probability=0.3,
                model_type="t-lite"
            )
        else:
            raise ValueError(f"Unknown LLM model type: {llm_model_type}")
        
        for node_id, ntype in node_types.items():
            if ntype == "L":
                llm_manager.init_agent(str(node_id))

    history.append({k: v.__dict__ for k, v in state.items()})

    # ---------- ОСНОВНОЙ ЦИКЛ (t от 1 до T_steps) ----------
    for t in range(1, T_steps + 1):
        # ----- 1. ГЕНЕРАЦИЯ НОВЫХ СООБЩЕНИЙ -----
        for node_id, ntype in node_types.items():
            if ntype == "B":
                continue
            if node_id in quarantined:
                continue
            gen_prob = P_USER if ntype == "U" else (P_RED if ntype == "R" else P_LLM)
            if random.random() < gen_prob:
                if ntype == "L" and llm_manager:
                    msg_data = llm_manager.generate_message(str(node_id))
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
                    # ---- ОТЛАДКА: красные генерируют ----
                    if ntype == "R" and verbose:
                        print(f"🔴 Red {node_id} generated at t={t}: {msg.text[:30]} (h={msg.h})")
                    frontier.append((node_id, msg, 0))
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

        # ----- 2. РАСПРОСТРАНЕНИЕ СООБЩЕНИЙ -----
        new_frontier = []
        reposted_this_step = set()
        used_edges_this_step = set()

        for edge, (_, release_step) in list(edge_busy.items()):
            if release_step <= t:
                del edge_busy[edge]

        for sender, msg, age in frontier:
            sender_type = node_types.get(sender, "U")
            if sender in quarantined:
                continue
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

                receiver_type = node_types.get(receiver, "U")

                # ----- СИНИЙ МОДЕРАТОР (ОЦЕНКА; БЛОКИРОВКА ТОЛЬКО ПРИ defense_policy) -----
                if receiver_type == "B" and not is_warning:
                    detected_risk, detected_category = blue_agent_detect(msg, blue_agent)
                    blue_failed = detected_risk is None
                    event = {
                        "t": t,
                        "from": sender,
                        "to": receiver,
                        "msg_id": msg.msg_id,
                        "text": msg.text,
                        "category": f"detected_{detected_category}",
                        "b": msg.b,
                        "h": msg.h,
                        "age": age,
                        "detected_risk": detected_risk,
                        "detected_category": detected_category,
                        "blue_failed": blue_failed,
                    }
                    # Активная защита: валидную высокую оценку → источник в карантин.
                    # Проваленные оценки (None) НЕ запускают защиту и исключаются из метрик.
                    if (
                        defense_policy is not None
                        and not blue_failed
                        and detected_risk >= defense_policy
                    ):
                        quarantined.add(msg.src)
                        event["quarantined_source"] = msg.src
                    timeline.append(event)
                    continue

                # ----- LLM ПОЛУЧАЕТ СООБЩЕНИЕ (история) -----
                if receiver_type == "L":
                    llm_manager.receive_message(str(receiver), msg.text)
                    timeline.append({
                        "t": t,
                        "from": sender,
                        "to": receiver,
                        "msg_id": msg.msg_id,
                        "text": msg.text,
                        "category": "received_by_llm",
                        "b": msg.b,
                        "h": msg.h,
                        "age": age,
                    })
                    # ---- ОТЛАДКА: LLM получил ----
                    if msg.h > 0.5 and verbose:
                        print(f"🟡 LLM {receiver} received HIGH-RISK msg from {sender}: {msg.text[:30]} (h={msg.h})")
                    continue   # LLM не репостит

                # ----- РЕПОСТ (только U и R) -----
                if receiver_type not in ("U", "R"):
                    continue

                rel = G[sender][receiver].get("weight", 1.0)
                k_val = kappa(state[receiver].b, msg.b, rp.alpha)
                p = sigmoid(
                    rp.lambda0 + rp.lambda1 * k_val +
                    rp.lambda2 * state[receiver].e +
                    rp.lambda3 * msg.h + rp.lambda4 * rel
                ) * (0.99 ** age)

                if random.random() < p:
                    if receiver_type == "U":
                        state[receiver], _ = update_user(state[receiver], msg, rp)
                    infected[msg.msg_id].add(receiver)
                    reposted_this_step.add(receiver)
                    used_edges_this_step.add(edge)
                    new_frontier.append((receiver, msg, age + 1))
                    timeline.append({
                        "t": t,
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

        frontier = new_frontier
        history.append({k: v.__dict__ for k, v in state.items()})

    # ---------- СБОР ДАННЫХ ДЛЯ LLM (визуализация) ----------
    llm_agent_data = {}
    if llm_manager:
        for node_id_str, hist in llm_manager.histories.items():
            node_id = int(node_id_str)
            received_msgs = hist.get("messages", [])
            generated = [
                e for e in timeline 
                if e.get("from") == node_id and e.get("category") not in ("warning", "detected_", "received_by_llm")
            ]
            llm_agent_data[node_id] = {
                "received": received_msgs,
                "generated": generated,
                "generation_count": hist.get("generation_count", 0)
            }

    risk_scores, risk_levels = calculate_risk_scores(timeline, state, node_types)
    formatted_history = [{str(nid): sd for nid, sd in snap.items()} for snap in history]

    return {
        "users_final": {k: v.__dict__ for k, v in state.items()},
        "timeline": timeline,
        "states_history": formatted_history,
        "total_messages": _msg_counter,
        "risk_scores": risk_scores,
        "risk_levels": risk_levels,
        "llm_agent_data": llm_agent_data,
        "quarantined": sorted(quarantined),
    }