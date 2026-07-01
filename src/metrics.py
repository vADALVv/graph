# metrics.py
# Классические диффузионные метрики + контрфактуал "с защитой / без".

from __future__ import annotations

from collections import defaultdict
from statistics import pvariance
from typing import Dict, Any, Optional, Callable, Tuple

# служебные категории, не являющиеся реальным распространением контента
_SERVICE_PREFIXES = ("detected_", "received_by_llm", "warning")


def _cat(ev: dict) -> str:
    return str(ev.get("category", ""))


def _is_seed(ev: dict) -> bool:
    """Исходная публикация (to is None) реального сообщения."""
    return ev.get("to") is None and not _cat(ev).startswith(_SERVICE_PREFIXES)


def _is_spread(ev: dict) -> bool:
    """Репост: есть получатель и это не служебное событие."""
    return (
        ev.get("to") is not None
        and ev.get("to") != -1
        and not _cat(ev).startswith(_SERVICE_PREFIXES)
    )


def _belief_of(state) -> float:
    return float(state["b"] if isinstance(state, dict) else getattr(state, "b", 0.0))


def compute_diffusion_metrics(
    results: Dict[str, Any],
    node_types: Dict[Any, str],
    belief_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Возвращает словарь метрик:
      - cascade_size_mean / max        : охват каскада (число уникальных узлов на сообщение)
      - cascade_depth_mean / max       : глубина каскада (макс. число хопов, age)
      - r0_per_seed                    : эффективный R0 = репосты / посеянные сообщения
      - mean_secondary_per_cascade     : среднее (охват - 1) по каскадам
      - peak_time / peak_new_infections: время до пика и величина пика
      - frac_radicalized               : доля U с |b| >= порога убеждения
      - frac_above_threshold           : доля U с b >= порога убеждения
      - polarization_var               : дисперсия b по пользователям U
    """
    timeline = results.get("timeline", [])
    users_final = results.get("users_final", {})

    seeds = [e for e in timeline if _is_seed(e)]
    spreads = [e for e in timeline if _is_spread(e)]

    # ---- каскады по сообщениям ----
    reach: Dict[int, set] = defaultdict(set)
    depth: Dict[int, int] = defaultdict(int)
    for e in seeds:
        reach[e["msg_id"]].add(e["from"])
    for e in spreads:
        mid = e["msg_id"]
        reach[mid].add(e["to"])
        depth[mid] = max(depth[mid], int(e.get("age", 0)))

    cascade_sizes = [len(v) for v in reach.values()]
    cascade_depths = [depth[m] for m in reach]

    n_seeds = len(seeds)
    n_spreads = len(spreads)

    r0_per_seed = (n_spreads / n_seeds) if n_seeds else 0.0
    mean_secondary = (
        sum(s - 1 for s in cascade_sizes) / len(cascade_sizes) if cascade_sizes else 0.0
    )

    # ---- время до пика ----
    new_per_t: Dict[int, int] = defaultdict(int)
    for e in seeds:
        new_per_t[e["t"]] += 1
    for e in spreads:
        new_per_t[e["t"]] += 1
    if new_per_t:
        peak_t = max(new_per_t, key=lambda k: new_per_t[k])
        peak_value = new_per_t[peak_t]
    else:
        peak_t, peak_value = None, 0

    # ---- мнения пользователей (только U) ----
    u_beliefs = []
    for nid, st in users_final.items():
        key = int(nid) if str(nid).lstrip("-").isdigit() else nid
        ntype = node_types.get(key, node_types.get(str(key)))
        if ntype == "U":
            u_beliefs.append(_belief_of(st))

    n_users = len(u_beliefs)
    frac_radicalized = (
        sum(1 for b in u_beliefs if abs(b) >= belief_threshold) / n_users
        if n_users else 0.0
    )
    frac_above = (
        sum(1 for b in u_beliefs if b >= belief_threshold) / n_users
        if n_users else 0.0
    )
    polarization_var = pvariance(u_beliefs) if n_users > 1 else 0.0

    return {
        "n_seeds": n_seeds,
        "n_spreads": n_spreads,
        "cascade_size_mean": (sum(cascade_sizes) / len(cascade_sizes)) if cascade_sizes else 0.0,
        "cascade_size_max": max(cascade_sizes) if cascade_sizes else 0,
        "cascade_depth_mean": (sum(cascade_depths) / len(cascade_depths)) if cascade_depths else 0.0,
        "cascade_depth_max": max(cascade_depths) if cascade_depths else 0,
        "r0_per_seed": r0_per_seed,
        "mean_secondary_per_cascade": mean_secondary,
        "peak_time": peak_t,
        "peak_new_infections": peak_value,
        "frac_radicalized": frac_radicalized,
        "frac_above_threshold": frac_above,
        "polarization_var": polarization_var,
        "n_users": n_users,
    }


def compute_blue_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Статистика модератора по событиям detected_*.
    Проваленные оценки (blue_failed=True или detected_risk is None) ИСКЛЮЧАЮТСЯ
    из расчёта среднего, но считаются отдельно.
    """
    timeline = results.get("timeline", [])
    detected = [e for e in timeline if _cat(e).startswith("detected_")]

    valid = [
        e for e in detected
        if not e.get("blue_failed") and e.get("detected_risk") is not None
    ]
    failed = [e for e in detected if e.get("blue_failed") or e.get("detected_risk") is None]

    risks = [float(e["detected_risk"]) for e in valid]
    quarantine_actions = sum(1 for e in timeline if "quarantined_source" in e)

    return {
        "assessed_total": len(detected),
        "assessed_valid": len(valid),
        "assessed_failed": len(failed),
        "avg_risk_valid": (sum(risks) / len(risks)) if risks else 0.0,
        "max_risk_valid": max(risks) if risks else 0.0,
        "quarantine_actions": quarantine_actions,
    }


def format_metrics(m: Dict[str, Any]) -> str:
    """Человекочитаемый вывод метрик."""
    pt = m.get("peak_time")
    return (
        f"   - Каскад (охват):     mean={m['cascade_size_mean']:.2f}, max={m['cascade_size_max']}\n"
        f"   - Каскад (глубина):   mean={m['cascade_depth_mean']:.2f}, max={m['cascade_depth_max']}\n"
        f"   - Эффективный R0:     {m['r0_per_seed']:.3f} (вторичных/посев)\n"
        f"   - Втор. на каскад:    {m['mean_secondary_per_cascade']:.3f}\n"
        f"   - Время до пика:      t={pt} ({m['peak_new_infections']} новых заражений)\n"
        f"   - Радикализация |b|≥τ: {m['frac_radicalized']*100:.1f}%\n"
        f"   - Доля b≥τ:           {m['frac_above_threshold']*100:.1f}%\n"
        f"   - Поляризация (var b): {m['polarization_var']:.4f}"
    )


def run_counterfactual(
    make_world: Callable[[], Tuple],
    sim_fn: Callable,
    sim_kwargs: Dict[str, Any],
    belief_threshold: float = 0.5,
    defense_policy: float = 0.6,
) -> Dict[str, Any]:
    """
    Контрфактуал "без защиты / с защитой".

    make_world() -> (G, users, node_types) — должен возвращать СВЕЖИЙ мир
                    на каждый вызов (иначе состояние перетечёт между прогонами).
    sim_fn       -> simulate_diffusion.
    defense_policy: порог риска, при котором модератор отправляет источник в карантин.

    Защита включается параметром simulate_diffusion(defense_policy=...).
    По умолчанию (defense_policy=None) модератор только наблюдает.
    """
    # --- без защиты ---
    G0, users0, nt0 = make_world()
    res0 = sim_fn(G=G0, users=users0, node_types=nt0, defense_policy=None, **sim_kwargs)
    m0 = compute_diffusion_metrics(res0, nt0, belief_threshold)

    # --- с защитой ---
    G1, users1, nt1 = make_world()
    res1 = sim_fn(G=G1, users=users1, node_types=nt1, defense_policy=defense_policy, **sim_kwargs)
    m1 = compute_diffusion_metrics(res1, nt1, belief_threshold)

    delta = {
        k: m1[k] - m0[k]
        for k in m0
        if isinstance(m0[k], (int, float)) and isinstance(m1.get(k), (int, float))
    }

    return {
        "no_defense": m0,
        "with_defense": m1,
        "delta": delta,
        "results_no_defense": res0,
        "results_with_defense": res1,
    }
