"""The committed Python-vs-JS fixture (tests/fixtures/engine_fixture.json) is what the Python code produces now."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import make_engine_fixture  # noqa: E402


def flat(x, out):
    if isinstance(x, dict):
        for k in sorted(x):
            out.append(k)
            flat(x[k], out)
    elif isinstance(x, list):
        out.append(len(x))
        for v in x:
            flat(v, out)
    else:
        out.append(x)
    return out


def test_fixture_is_current():
    committed = flat(json.loads(make_engine_fixture.OUT.read_text()), [])
    fresh = flat(json.loads(json.dumps(make_engine_fixture.build())), [])
    assert len(committed) == len(fresh), "run scripts/make_engine_fixture.py"
    for a, b in zip(committed, fresh):
        if isinstance(a, float) or isinstance(b, float):
            assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), "run scripts/make_engine_fixture.py"
        else:
            assert a == b, "run scripts/make_engine_fixture.py"
