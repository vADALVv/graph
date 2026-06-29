from __future__ import annotations

import random
from typing import Tuple, Dict
import networkx as nx
from dataclasses import dataclass


@dataclass
class UserState:
    b: float
    c: float
    e: float


SEED = 42
random.seed(SEED)


def create_graph(
    num_u: int,
    num_r: int,
    num_l: int,
    num_b: int = 0,
    avg_degree: int = 4
) -> Tuple[nx.DiGraph, Dict[int, UserState], Dict[int, str]]:

    if num_u <= 0:
        raise ValueError("num_u must be > 0")

    # -------------------------
    # USER GRAPH (только U и R)
    # -------------------------
    k = max(2, int(avg_degree * 0.7))
    if k % 2 != 0:
        k += 1
    if k >= num_u:
        k = max(1, num_u - 1)
        if k % 2 != 0 and k > 1:
            k -= 1

    G_u = nx.newman_watts_strogatz_graph(
        n=num_u,
        k=min(k, num_u - 1),
        p=0.3,
        seed=SEED
    )
    G = G_u.to_directed()

    users: Dict[int, UserState] = {}
    roles: Dict[int, str] = {}

    # Пользователи (U)
    for node in G.nodes():
        users[node] = UserState(
            b=random.uniform(-1, 1),
            c=random.uniform(0.3, 0.9),
            e=random.uniform(-1, 1)
        )
        roles[node] = "U"

    next_id = len(G.nodes())
    base_nodes = list(G.nodes())

    # Красные узлы (R)
    for _ in range(num_r):
        G.add_node(next_id)
        roles[next_id] = "R"
        users[next_id] = UserState(
            b=random.uniform(-1, 1),
            c=random.uniform(0.3, 0.9),
            e=random.uniform(-1, 1)
        )
        if base_nodes:
            targets = random.sample(base_nodes, min(len(base_nodes), avg_degree * 2))
            for t in targets:
                G.add_edge(next_id, t, weight=random.uniform(0.5, 1.0))
        next_id += 1

    # LLM узлы (L) – не имеют состояния, ставим нули
    for _ in range(num_l):
        G.add_node(next_id)
        roles[next_id] = "L"
        users[next_id] = UserState(b=0.0, c=0.0, e=0.0)   # нет состояния
        if base_nodes:
            targets = random.sample(base_nodes, min(len(base_nodes), avg_degree))
            for t in targets:
                G.add_edge(next_id, t, weight=random.uniform(0.5, 1.0))
        next_id += 1

    # Синие узлы-модераторы (B) – не имеют состояния
    for _ in range(num_b):
        G.add_node(next_id)
        roles[next_id] = "B"
        users[next_id] = UserState(b=0.0, c=0.0, e=0.0)   # нет состояния
        if base_nodes:
            targets = random.sample(base_nodes, min(len(base_nodes), avg_degree * 3))
            for t in targets:
                G.add_edge(next_id, t, weight=1.0)        # сильная связь для мониторинга
        # связи между модераторами
        blue_nodes_start = next_id - num_b
        for existing_b in range(blue_nodes_start, next_id):
            if existing_b >= 0 and existing_b != next_id:
                G.add_edge(next_id, existing_b, weight=0.8)
                G.add_edge(existing_b, next_id, weight=0.8)
        next_id += 1

    print(f"✅ Graph created with {next_id} nodes:")
    print(f"   - Users (U): {num_u}")
    print(f"   - Red agents (R): {num_r}")
    print(f"   - LLM agents (L): {num_l}")
    print(f"   - Blue moderators (B): {num_b}")

    return G, users, roles

