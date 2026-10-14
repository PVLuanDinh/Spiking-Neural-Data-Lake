"""C1: restart tests get a bounded child deadline in an ordinary pytest invocation."""

import os
import subprocess
import sys
import time

from conftest import STAGE_DEADLINE_S, bounded_stage_deadline


def test_session_start_generated_a_future_bounded_deadline():
    remaining = (int(os.environ["SNDL_R2_CHILD_DEADLINE_NS"]) - time.time_ns()) / 1e9
    assert 0 < remaining <= STAGE_DEADLINE_S


def test_a_stricter_caller_deadline_is_respected_and_a_looser_one_is_capped():
    now = 10**18
    strict = str(now + 60 * 10**9)
    assert bounded_stage_deadline(now, strict) == strict
    assert bounded_stage_deadline(now, str(now + 10**15 * 10**9)) == str(
        now + STAGE_DEADLINE_S * 10**9
    )
    assert bounded_stage_deadline(now, None) == bounded_stage_deadline(now, "garbage")


def test_an_expired_caller_deadline_is_ignored_and_a_future_one_still_wins():
    """Board #2566: a caller value <= now is ignored (own is used). A caller value in the
    future, even 1 ns, is still respected as stricter -- that is the spec, not widened here."""
    now = 10**18
    own = str(now + STAGE_DEADLINE_S * 10**9)
    assert bounded_stage_deadline(now, str(now - 1)) == own
    assert bounded_stage_deadline(now, str(now)) == own
    assert bounded_stage_deadline(now, str(now + 1)) == str(now + 1)


def test_an_expired_caller_env_deadline_does_not_reach_the_children():
    env = {**os.environ, "SNDL_R2_CHILD_DEADLINE_NS": "1000000000"}
    node = "tests/test_stage_deadline.py::test_session_start_generated_a_future_bounded_deadline"
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-cov", node]
    root = os.path.dirname(os.path.dirname(__file__))
    assert subprocess.run(cmd, env=env, cwd=root).returncode == 0
