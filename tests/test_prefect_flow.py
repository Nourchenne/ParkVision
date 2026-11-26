import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("prefect")

FLOW_PATH = Path(__file__).resolve().parents[1] / "orchestration" / "parkvision_flow.py"


def load_flow_module():
    spec = importlib.util.spec_from_file_location("parkvision_flow", FLOW_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dvc_command_targets_one_stage_with_current_python():
    module = load_flow_module()
    assert module.dvc_command("prepare_split") == [
        sys.executable, "-m", "dvc", "repro", "prepare_split",
    ]


def test_run_stage_runs_from_project_root_and_fails_loudly():
    module = load_flow_module()
    with mock.patch.object(module.subprocess, "run") as run:
        module.run_stage("prepare_yolo")
    run.assert_called_once_with(
        module.dvc_command("prepare_yolo"), cwd=module.PROJECT_ROOT, check=True,
    )


def test_tasks_map_to_the_two_dvc_stages():
    module = load_flow_module()
    with mock.patch.object(module, "run_stage") as run_stage:
        module.run_prepare_split.fn()
        module.run_prepare_yolo.fn()
    assert [call.args[0] for call in run_stage.call_args_list] == [
        "prepare_split", "prepare_yolo",
    ]


def test_flow_runs_split_before_yolo():
    module = load_flow_module()
    order = []
    with mock.patch.object(module, "run_prepare_split", side_effect=lambda: order.append("split")), \
            mock.patch.object(module, "run_prepare_yolo", side_effect=lambda: order.append("yolo")):
        module.parkvision_pipeline.fn()
    assert order == ["split", "yolo"]


def test_dvc_yaml_defines_the_stages_the_flow_calls():
    import yaml

    dvc_yaml = yaml.safe_load((FLOW_PATH.parents[1] / "dvc.yaml").read_text(encoding="utf-8"))
    assert {"prepare_split", "prepare_yolo"} <= set(dvc_yaml["stages"])
