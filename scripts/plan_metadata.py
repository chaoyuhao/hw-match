"""Versioned runner contract. Geometry checks are not a second tile selector."""
import math
import re

PROBLEM_FIELDS = ('b', 'm', 'n', 'k', 'dtype', 'ta', 'tb')


def problem(case):
    return {key: case[key] for key in PROBLEM_FIELDS}


def tile_token(token):
    if not isinstance(token, str) or not re.fullmatch(r'[1-9][0-9]*x[1-9][0-9]*', token):
        raise ValueError('invalid tile token')
    m, n = map(int, token.split('x'))
    if any(x < 16 or x > 256 or x % 16 for x in (m, n)):
        raise ValueError('tile outside planner search bounds')
    return m, n


def validate(plan, case):
    if not isinstance(plan, dict) or plan.get('schema_version') != 2 or plan.get('planner_version') != 1:
        raise ValueError('unsupported Matmul plan schema/planner version')
    if plan.get('problem') != problem(case):
        raise ValueError('Matmul plan input metadata mismatch')
    fields = ('tile_m', 'tile_n', 'tasks', 'blocks', 'available_cores', 'ub_bytes', 'ub_budget')
    if any(type(plan.get(k)) is not int or plan[k] < 1 for k in fields):
        raise ValueError('invalid Matmul counts/resources')
    if plan['available_cores'] > 65535:
        raise ValueError('invalid core count')
    token = f"{plan['tile_m']}x{plan['tile_n']}"
    m, n = tile_token(token)
    if plan.get('policy') not in ('auto', token) or plan.get('plan_id') != f'gm-v1-{token}':
        raise ValueError('Matmul plan identity/request mismatch')
    tasks = case['b'] * ((case['m'] + m - 1) // m) * ((case['n'] + n - 1) // n)
    if plan['tasks'] != tasks or plan['blocks'] != min(tasks, plan['available_cores']):
        raise ValueError('Matmul plan grid mismatch')
    if plan['ub_budget'] != min(128*1024, plan['ub_bytes'] - 64*1024):
        raise ValueError('Matmul UB budget mismatch')
    inner = plan.get('inner_tile')
    if not isinstance(inner, dict) or set(inner) != {'m','n','k'} or any(type(v) is not int or v <= 0 for v in inner.values()):
        raise ValueError('missing or invalid SDK inner tile')
    score = plan.get('heuristic_score')
    if type(score) not in (int, float) or not math.isfinite(score) or score <= 0:
        raise ValueError('invalid heuristic score')
    return plan


def validate_discovery(data, case):
    if not isinstance(data, dict) or data.get('schema_version') != 2:
        raise ValueError('unsupported discovery schema')
    auto = validate(data.get('auto'), case)
    if auto['policy'] != 'auto':
        raise ValueError('discovery auto plan has wrong request')
    entries = data.get('candidates')
    if not isinstance(entries, list) or not 1 <= len(entries) <= 64:
        raise ValueError('invalid candidate count')
    seen, accepted = set(), {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('invalid discovery entry')
        token = entry.get('policy')
        tile_token(token)
        if token in seen or type(entry.get('accepted')) is not bool:
            raise ValueError('duplicate or invalid candidate entry')
        seen.add(token)
        if entry['accepted']:
            plan = validate(entry.get('plan'), case)
            if plan['policy'] != token or any(plan[k] != auto[k] for k in ('available_cores', 'ub_bytes')):
                raise ValueError('candidate request/hardware mismatch')
            accepted[token] = plan
    actual = f"{auto['tile_m']}x{auto['tile_n']}"
    if actual not in accepted or {**accepted[actual], 'policy':'auto'} != auto:
        raise ValueError('automatic plan missing or inconsistent in discovered candidates')
    return accepted
