from __future__ import annotations

import sys

import pytest


@pytest.mark.parametrize(
    ("module_name", "entrypoint", "selector", "workflow_module"),
    [
        ("geo2wf.cli.infer", "main", "deterministic-residual", "run_storm_unet_inference"),
        ("geo2wf.cli.evaluate", "main", "intensity-correction", "evaluate_intensity_correction"),
        ("geo2wf.cli.export", "main", "geo-sar", "export_geo_sar_geotiffs"),
    ],
)
def test_dispatch_forwards_workflow_help(monkeypatch, module_name, entrypoint, selector, workflow_module):
    module = __import__(module_name, fromlist=[entrypoint])
    called = {}

    def fake_main():
        called["argv"] = list(sys.argv)

    monkeypatch.setattr(
        __import__(f"geo2wf.workflows.{workflow_module}", fromlist=["main"]),
        "main",
        fake_main,
    )
    monkeypatch.setattr(sys, "argv", ["geo2wf", selector, "--help"])
    getattr(module, entrypoint)()
    assert called["argv"] == ["geo2wf", "--help"]
