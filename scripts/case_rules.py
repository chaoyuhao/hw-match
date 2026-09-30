"""Reproducible coverage rules, independent of the unknown online test shapes."""
import hashlib
import json
import random

FIELDS = ('b', 'm', 'n', 'k', 'dtype', 'ta', 'tb', 'pattern', 'seed')
FAMILIES = ('compact', 'aligned', 'tails', 'tall', 'wide', 'core_boundary', 'long_k', 'batch_boundary')


def identity(case):
    return tuple(case[key] for key in FIELDS)


def memory_bound(c):
    # Conservative allowance for packed inputs, FP64 golden and temporaries.
    return 32 * c['b'] * (c['m']*c['k'] + c['k']*c['n'] + c['m']*c['n'] + 1)


def generate_cases(count=64, seed=20260930, cores=24):
    if not 1 <= count <= 4096 or seed < 0 or not 1 <= cores <= 65535:
        raise ValueError('invalid generated case count/seed/core hint')
    result = []
    for i in range(count):
        family = FAMILIES[(i // 8) % len(FAMILIES)]
        rng = random.Random(seed * 4096 + i)
        for attempt in range(128):
            tile = 16 * (2 ** rng.randrange(5))
            b, m, n, k = 1, tile, tile, 8 * (2 ** rng.randrange(6))
            if family == 'compact': m, n = rng.randint(1,32), rng.randint(1,32)
            elif family == 'aligned': n = 16 * rng.randint(1,16)
            elif family == 'tails': m, n = tile + rng.choice((-1,1)), 16*rng.randint(1,16)+rng.choice((-1,1))
            elif family == 'tall': m, n, k = 2**rng.randrange(9,13)+rng.choice((-1,0,1)), rng.randint(1,33), 8*2**rng.randrange(4)
            elif family == 'wide': n, m, k = 2**rng.randrange(9,13)+rng.choice((-1,0,1)), rng.randint(1,33), 8*2**rng.randrange(4)
            elif family == 'core_boundary': m, n, k = min(8192, cores*16+rng.choice((-1,0,1))), rng.randint(1,32), 8*2**rng.randrange(4)
            elif family == 'long_k': m, n, k = 16*rng.randint(1,4), 16*rng.randint(1,4), 2**rng.randrange(9,14)
            elif family == 'batch_boundary': b, m, n, k = max(1,min(257,cores+rng.choice((-1,0,1)))), rng.randint(1,32), rng.randint(1,32), 8*2**rng.randrange(4)
            case = dict(b=b,m=m,n=n,k=k,dtype='fp16' if i%8 < 4 else 'bf16',
                        ta=bool((i%4)//2),tb=bool(i%2),pattern='negative' if i%3==0 else 'random',
                        seed=seed*4096+i)
            if b*m*n*k <= 32*1024*1024 and memory_bound(case) <= 128*1024*1024:
                break
        else:
            raise ValueError('generation budget exhausted')
        case_id = hashlib.sha256(json.dumps(identity(case), separators=(',', ':')).encode()).hexdigest()[:20]
        case.update(case_id=case_id, name=f'rule_{family}_{case_id}',
                    generation=dict(version=1, family=family, seed=seed, index=i, cores_hint=cores, attempt=attempt))
        result.append(case)
    return result


def add_arguments(parser):
    parser.add_argument('--case-count', type=int, default=64, help='generated suite count (default 64)')
    parser.add_argument('--case-seed', type=int, default=20260930, help='reproducible shape/input seed')
    parser.add_argument('--case-cores', type=int, default=24, help='generation boundary hint; never overrides runtime cores')


def from_arguments(args, parser):
    try:
        return generate_cases(args.case_count, args.case_seed, args.case_cores)
    except ValueError as error:
        parser.error(str(error))
