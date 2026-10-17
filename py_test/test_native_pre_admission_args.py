"""The installed Python entry point must pass the opt-in into native config."""

import subprocess
import sys

import pytest

from vllm_router.launch_router import parse_router_args
from vllm_router.router_args import RouterArgs
from vllm_router_rs import Router


def test_native_retry_cli_is_default_off_and_passes_one_to_real_extension():
    assert parse_router_args([]).native_pre_admission_retries == 0
    selected = parse_router_args(["--native-pre-admission-retries", "1"])
    assert selected.native_pre_admission_retries == 1
    assert Router(worker_urls=[], native_pre_admission_retries=1) is not None
    with pytest.raises(SystemExit):
        parse_router_args(["--native-pre-admission-retries", "2"])


@pytest.mark.parametrize("value", [-1, 2, True, "1", 1.0])
def test_native_retry_python_args_reject_invalid_policy(value):
    with pytest.raises(ValueError, match="must be 0 or 1"):
        RouterArgs(native_pre_admission_retries=value)


@pytest.mark.parametrize("mode", [{"mini_lb": True}, {"vllm_pd_disaggregation": True}])
def test_native_retry_python_args_reject_unsupported_router_modes(mode):
    with pytest.raises(ValueError, match="require regular Rust routing"):
        RouterArgs(native_pre_admission_retries=1, **mode)


def test_real_extension_projects_policy_into_native_validation_before_startup():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vllm_router_rs import Router; "
            "Router(worker_urls=[],native_pre_admission_retries=2).start()",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0
    assert "native_pre_admission_retries must be 0 or 1" in result.stderr
