import importlib
import sys
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("prefect")

ORCHESTRATION = str(Path(__file__).resolve().parents[1] / "orchestration")


def load_serve_module():
    sys.path.insert(0, ORCHESTRATION)
    try:
        return importlib.import_module("serve_pipeline")
    finally:
        sys.path.remove(ORCHESTRATION)


def run_main(module, *args):
    with mock.patch.object(module.parkvision_pipeline, "serve") as serve, \
            mock.patch.object(sys, "argv", ["serve_pipeline.py", *args]):
        module.main()
    return serve


def test_deployment_is_named_and_scheduled_nightly_by_default():
    module = load_serve_module()
    serve = run_main(module)
    serve.assert_called_once_with(name="parkvision-nightly", cron="0 3 * * *", tags=["data", "dvc"])


def test_no_schedule_serves_a_manual_only_deployment():
    module = load_serve_module()
    serve = run_main(module, "--no-schedule")
    assert serve.call_args.kwargs["cron"] is None
    assert serve.call_args.kwargs["name"] == "parkvision-nightly"


def test_custom_cron_is_passed_through():
    module = load_serve_module()
    serve = run_main(module, "--cron", "*/30 * * * *")
    assert serve.call_args.kwargs["cron"] == "*/30 * * * *"
