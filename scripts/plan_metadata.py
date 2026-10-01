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


def validate_execution(plan, case):
    """Actual family contract; no timing thresholds or second auto selector."""
    if not isinstance(plan, dict) or plan.get('schema_version') != 3:
        raise ValueError('unsupported execution schema')
    if plan.get('problem') != problem(case):
        raise ValueError('execution input metadata mismatch')
    family, requested = plan.get('family'), plan.get('requested_family')
    if family not in ('gm', 'small', 'stream', 'pipeline') or requested not in ('auto', 'gm', 'small', 'stream', 'pipeline'):
        raise ValueError('invalid execution family')
    if requested != 'auto' and requested != family:
        raise ValueError('requested and actual family differ')
    if type(plan.get('similarity_available')) is not bool or plan['similarity_available'] != (family == 'gm'):
        raise ValueError('similarity availability does not match family')
    for key in ('tasks', 'blocks', 'available_cores', 'ub_bytes'):
        if type(plan.get(key)) is not int or plan[key] < 1:
            raise ValueError('invalid execution counts/resources')
    if plan['available_cores'] > 65535 or plan['blocks'] != min(plan['tasks'], plan['available_cores']):
        raise ValueError('invalid execution blocks')
    requested_sum = plan.get('requested_sum', 'auto')
    if requested_sum not in ('auto','r15','rows','partials') or (family == 'small' and requested_sum not in ('auto','r15')):
        raise ValueError('invalid or conflicting reduction request')
    selection = plan.get('selection')
    if selection is not None:
        if (family == 'small' or not isinstance(selection,dict) or
            type(selection.get('planner_version')) is not int or selection['planner_version'] not in (2,3) or
            selection.get('cost_model') != {2:'joint-work-v1',3:'joint-work-v2'}[selection['planner_version']] or
            type(selection.get('score')) not in (int,float) or not math.isfinite(selection['score']) or selection['score'] <= 0):
            raise ValueError('invalid joint selection metadata')
    if family == 'gm':
        gm = validate(plan.get('matmul'), case)
        validate_reduction(plan,case,32,47392,gm['ub_budget'])
        if plan.get('variant') != 'mix' or any(plan[k] != gm[k] for k in ('tasks','blocks','available_cores','ub_bytes')):
            raise ValueError('GM execution disagrees with Matmul plan')
    elif family in ('stream','pipeline'):
        validate_stream(plan, case)
    else:
        b, m, n, k = (case[key] for key in ('b','m','n','k'))
        dot = (not case['ta'] or m == 1) and (case['tb'] or n == 1)
        if not (1 <= b <= (2**32-1)*8 and 1 <= m <= 8192 and 1 <= n <= 64 and 8 <= k <= 256
                and k % 8 == 0 and case['dtype'] in ('fp16','bf16')):
            raise ValueError('small geometry outside legal bounds')
        if plan.get('variant') != ('dot' if dot else 'rows') or (not dot and case['tb']):
            raise ValueError('small variant/layout mismatch')
        align = lambda x: (x + 15) // 16 * 16
        a_elements = m*align(k) if dot or not case['ta'] else k*align(m)
        b_elements = n*align(k) if dot else k*align(n)
        legacy_used = 6*(a_elements+b_elements)+1344
        columns = plan.get('dot_columns', 0)  # Older schema-3 reports use the original Dot.
        if type(columns) is not int or columns < 0:
            raise ValueError('invalid batched Dot column count')
        if columns and (not dot or not 2 <= columns <= n or (columns != n and columns % 8)):
            raise ValueError('batched Dot column alignment/layout mismatch')
        used = (6*(a_elements+b_elements)+4*(columns*64+(columns+7)//8*8)+288
                if columns else legacy_used)
        limit = min(65536,plan['ub_bytes']-32768)
        if type(plan.get('ub_used')) is not int or plan['ub_used'] != used or max(used,legacy_used) > limit:
            raise ValueError('small resource accounting mismatch')
        if plan['tasks'] != (b+7)//8 or 'matmul' in plan or 'reduction' in plan:
            raise ValueError('small tasks or unexpected Matmul metadata')
    validate_reduction_expansion(plan, case)
    return plan


def validate_stream(plan, case):
    """Validate emitted ownership/storage, not whether Auto chose an optimal plan."""
    s = plan.get('stream')
    pipeline = plan['family'] == 'pipeline'
    if (not isinstance(s, dict) or s.get('planner_version') not in (1,2,3) or
        plan.get('variant') != ('mix_pipeline' if pipeline else 'mix_stream') or 'matmul' in plan):
        raise ValueError('invalid stream plan identity')
    version = s['planner_version']
    buffers = s.get('buffers', 1 if version == 1 else None)
    if (type(buffers) is not int or buffers != (2 if pipeline else 1) or
        (pipeline and version not in (2,3)) or (version >= 2 and not plan.get('selection'))):
        raise ValueError('stream buffering/selection mismatch')
    fields = ('tile_m','tile_n','splits','row_pitch','c_slot_elements','maxima_offset',
              'partial_offset','scratch_bytes','max_tiles_per_core','ub_used','ub_budget')
    if any(type(s.get(key)) is not int or s[key] < 1 for key in fields):
        raise ValueError('invalid stream counts/resources')
    tm, tn = tile_token(f"{s['tile_m']}x{s['tile_n']}")
    b, m, n, k = (case[key] for key in ('b','m','n','k'))
    if not (b > 0 and all(1 <= x <= 8192 for x in (m,n,k)) and case['dtype'] in ('fp16','bf16')):
        raise ValueError('invalid stream problem')
    columns, splits = (n+tn-1)//tn, s['splits']
    tasks = b*((m+tm-1)//tm)*splits
    if splits > columns or plan['tasks'] != tasks or tasks > 2**64-1:
        raise ValueError('invalid or empty stream task grid')
    pitch = (m+31)//32*32
    slot = min(m,tm)*tn
    maxima = plan['blocks']*buffers*slot*4
    base_ub = 32768+tm*4+128+(256 if splits>1 else 0)+14368
    reduction = validate_reduction(plan,case,tm if splits==1 else 32,base_ub,s['ub_budget'])
    if reduction['mode'] == 'partials' and version != 3:
        raise ValueError('partial records require stream planner v3')
    if version == 3 and plan.get('selection',{}).get('planner_version') != 3:
        raise ValueError('stream and joint planner version mismatch')
    partial = maxima + (reduction['bytes'] if splits > 1 else 0)
    size = partial + (b*pitch*4*splits if splits>1 else reduction['bytes'])
    expected = dict(row_pitch=pitch,c_slot_elements=slot,maxima_offset=maxima,
                    partial_offset=partial,scratch_bytes=size,
                    ub_used=base_ub+reduction['fold_ub_bytes'],
                    ub_budget=min(128*1024,plan['ub_bytes']-64*1024))
    if size > 2**64-1 or any(s[key] != value for key,value in expected.items()):
        raise ValueError('stream allocation mismatch')
    if s['ub_used'] > 64*1024 or s['ub_used']+s['ub_budget'] > plan['ub_bytes']:
        raise ValueError('stream UB budget exceeded')
    blocks = plan['blocks']
    period = splits // math.gcd(blocks,splits)
    load = 0
    for core in range(min(blocks,splits)):
        count = (tasks-1-core)//blocks+1
        cycles, tail = divmod(count,period)
        weights = [columns*(((core+i*blocks)%splits)+1)//splits -
                   columns*((core+i*blocks)%splits)//splits for i in range(min(count,period))]
        load = max(load,cycles*sum(weights)+sum(weights[:tail]))
    if s['max_tiles_per_core'] != load:
        raise ValueError('stream per-core load mismatch')
    inner = s.get('inner_tile')
    if not isinstance(inner,dict) or set(inner) != {'m','n','k'} or any(type(v) is not int or v <= 0 for v in inner.values()):
        raise ValueError('missing or invalid stream SDK inner tile')


def validate_reduction(plan, case, span, base_ub, matmul_budget):
    """Validate record ownership and resource arithmetic; never re-run the selector."""
    r = plan.get('reduction')
    requested = plan.get('requested_sum','auto')
    selection_version = (plan.get('selection') or {}).get('planner_version',0)
    row_bytes = case['b']*((case['m']+31)//32)*128
    if r is None:
        if 'requested_sum' in plan or selection_version == 3:
            raise ValueError('missing reduction plan')
        return dict(mode='rows',bytes=row_bytes,fold_ub_bytes=0)
    if not isinstance(r,dict) or type(r.get('version')) is not int or r['version'] != 1 or r.get('mode') not in ('rows','partials'):
        raise ValueError('invalid reduction identity')
    if requested not in ('auto','r15') and requested != r['mode']:
        raise ValueError('requested and actual reduction differ')
    partial = r['mode'] == 'partials'
    if partial and selection_version != 3:
        raise ValueError('partial records require joint planner v3')
    capacity = 8
    while capacity < span: capacity *= 2
    count = (case['m']+span-1)//span
    extra = 14*capacity+64 if partial else 0
    expected = dict(segment_rows=span,segments=count,record_floats=16 if partial else 0,
                    bytes=case['b']*count*64 if partial else row_bytes,
                    fold_ub_bytes=extra,ub_used=base_ub+extra)
    if any(type(r.get(key)) is not int or r[key] != value for key,value in expected.items()):
        raise ValueError('reduction geometry/storage mismatch')
    if not 0 < r['bytes'] <= 2**64-1 or r['ub_used'] > 65536 or r['ub_used']+matmul_budget > plan['ub_bytes']:
        raise ValueError('reduction resource budget exceeded')
    return r


def validate_reduction_expansion(plan, case):
    """Check the frozen-plan trace; model ranking itself stays in the C++ planner."""
    selection = plan.get('selection') or {}
    requested = plan.get('requested_sum','auto')
    unrestricted = (plan['family'] != 'small' and plan['requested_family'] == 'auto' and
                    selection.get('planner_version') == 3 and
                    (plan['family'] != 'gm' or plan['matmul']['policy'] == 'auto'))
    if 'reduction_expansion' not in plan:
        if requested == 'r15' and unrestricted:
            raise ValueError('missing R15 expansion control trace')
        return  # Historical Auto reports predate the coverage experiment.
    trace = plan['reduction_expansion']
    if (not unrestricted or requested not in ('auto','r15') or not isinstance(trace,dict) or
        type(trace.get('version')) is not int or trace['version'] != 1 or
        type(trace.get('enabled')) is not bool or type(trace.get('applied')) is not bool or
        trace['enabled'] != (requested == 'auto') or trace.get('baseline_mode') not in ('rows','partials') or
        type(trace.get('baseline_score')) not in (int,float) or not math.isfinite(trace['baseline_score']) or trace['baseline_score'] <= 0):
        raise ValueError('invalid reduction expansion trace')
    reduction = plan.get('reduction')
    if not isinstance(reduction,dict):
        raise ValueError('expansion trace requires reduction metadata')
    if trace['applied']:
        if (not trace['enabled'] or trace['baseline_mode'] != 'rows' or reduction['mode'] != 'partials' or
            reduction['segments'] < 2 or reduction['segments']*16 >= case['m'] or
            selection['score'] < trace['baseline_score']):
            raise ValueError('invalid or non-compressing reduction expansion')
    elif trace['baseline_mode'] != reduction['mode'] or trace['baseline_score'] != selection['score']:
        raise ValueError('unchanged reduction disagrees with baseline trace')
