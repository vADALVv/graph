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
    avg_degree: int = 4,
    seed: int = SEED,
    verbose: bool = True
) -> Tuple[nx.DiGraph, Dict[int, UserState], Dict[int, str]]:

    if num_u <= 0:
        raise ValueError("num_u must be > 0")

    # каждый вызов с новым seed -> новый граф и новые состояния узлов
    random.seed(seed)

    # -------------------------
    # Базовый граф пользователей (только U)
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
        seed=seed
    )
    G = G_u.to_directed()   # U ↔ U (двунаправленные)

    users: Dict[int, UserState] = {}
    roles: Dict[int, str] = {}

    # Пользователи (U) — входящие и исходящие (уже есть)
    for node in G.nodes():
        users[node] = UserState(
            b=random.uniform(-1, 1),
            c=random.uniform(0.3, 0.9),
            e=random.uniform(-1, 1)
        )
        roles[node] = "U"

    next_id = len(G.nodes())
    base_nodes = list(G.nodes())

    # -------------------------
    # Красные узлы (R) — только исходящие
    # -------------------------
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
                # R → t (только исходящие)
                G.add_edge(next_id, t, weight=random.uniform(0.5, 1.0))
        next_id += 1

    # -------------------------
    # LLM узлы (L) — входящие и исходящие
    # -------------------------
    for _ in range(num_l):
        G.add_node(next_id)
        roles[next_id] = "L"
        users[next_id] = UserState(b=0.0, c=0.0, e=0.0)
        if base_nodes:
            targets = random.sample(base_nodes, min(len(base_nodes), avg_degree))
            for t in targets:
                # L → t (исходящие)
                G.add_edge(next_id, t, weight=random.uniform(0.5, 1.0))
                # t → L (входящие)
                G.add_edge(t, next_id, weight=random.uniform(0.5, 1.0))
        next_id += 1

    # -------------------------
    # Синие узлы (B) — только входящие
    # -------------------------
    for _ in range(num_b):
        G.add_node(next_id)
        roles[next_id] = "B"
        users[next_id] = UserState(b=0.0, c=0.0, e=0.0)
        if base_nodes:
            targets = random.sample(base_nodes, min(len(base_nodes), avg_degree * 3))
            for t in targets:
                # t → B (только входящие)
                G.add_edge(t, next_id, weight=1.0)
        # связи между модераторами (тоже только входящие для нового узла)
        blue_nodes_start = next_id - num_b
        for existing_b in range(blue_nodes_start, next_id):
            if existing_b >= 0 and existing_b != next_id:
                # existing_b → next_id (входящие для next_id)
                G.add_edge(existing_b, next_id, weight=0.8)
        next_id += 1

    if verbose:
        print(f"✅ Graph created with {next_id} nodes:")
        print(f"   - Users (U): {num_u}")
        print(f"   - Red agents (R): {num_r}")
        print(f"   - LLM agents (L): {num_l}")
        print(f"   - Blue moderators (B): {num_b}")

    return G, users, roles