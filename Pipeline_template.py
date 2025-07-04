#!/usr/bin/env python
"""
pipeline_template_v1.py – Vertex AI pipeline launcher
––––––––––––––––––––––––––––––––––––––––––––––––––––––
?? Changes in this version
    •  Robust `_parse_list()` so `--parallel_parameter_values`
       can contain items with spaces (e.g. "PUBLIC SECTOR").
    •  `_safe_name()` makes every display-name Vertex-safe.
    •  Fan-out jobs keep the *real* vertical string (with spaces),
       while their display-names are slug-ified.
Everything else is kept from the original template.
"""

import argparse
import builtins
import csv
import io
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.cloud import aiplatform, storage
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from kfp import dsl
import kfp
from kfp.registry import RegistryClient


def _parse_list(raw: str) -> List[str]:
    """Return a clean list of strings from almost anything."""
    if not raw:
        return []

    # 1) JSON list?
    try:
        val = json.loads(raw)
        if isinstance(val, list):
            return [str(x).strip(" \t\r\n'\"") for x in val]
    except json.JSONDecodeError:
        pass

    # 2) CSV / pipe / semicolon separated
    for delim in (",", "|", ";"):
        parsed = next(csv.reader([raw], delimiter=delim, skipinitialspace=True))
        if len(parsed) > 1:
            return [s.strip(" \t\r\n'\"") for s in parsed if s.strip()]

    # 3) shell-like split e.g. INDUSTRIAL "PUBLIC SECTOR"
    return [tok.strip(" \t\r\n'\"") for tok in shlex.split(raw)]


_name_pat = re.compile(r"[^A-Za-z0-9\-]+")
def _safe_name(text: str) -> str:
    """Make a legal Vertex display-name (<=128 chars, A-Z a-z 0-9 -)."""
    return _name_pat.sub("-", text)[:128]
# ?---------------------------------------------------------------?


def ensure_bucket(uri: str, project: str, location: str) -> None:
    name = uri.replace("gs://", "").rstrip("/")
    client = storage.Client(project=project)
    if client.lookup_bucket(name) is None:
        client.create_bucket(name, location=location)
        print(f"  Created bucket {uri}")
    else:
        print(f"  Bucket {uri} already exists – skipping.")


def ensure_artifact_repo(repo: str, project: str, location: str, desc: str = "") -> None:
    if subprocess.run(
        ["gcloud", "artifacts", "repositories", "describe", repo,
         "--project", project, "--location", location],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0:
        print(f"  Artifact repo {repo} already exists – skipping.")
        return
    subprocess.check_call(
        ["gcloud", "artifacts", "repositories", "create", repo,
         "--repository-format=docker", "--location", location,
         f"--description={desc or 'Vertex AI pipeline images'}", "--project", project]
    )
    print(f"  Created Artifact Registry repo {repo}")


def _worker_pool(image_uri: str, module_name: str, args_json: str, app_environment, machine_type: str,
                 second_arg: Optional[str] = None, accelerator_type: str = "",
                 accelerator_count: int = 0) -> List[Dict[str, Any]]:
    """Return Vertex AI workerPoolSpecs list (single replica)."""
    cli = [args_json]
    if second_arg:
        cli.append(second_arg)
    if app_environment:
        cli.append(app_environment)

    container = {"image_uri": image_uri,
                 "command": ["python", "-u", "-m", module_name],
                 "args": cli}
    machine = {"machine_type": machine_type}

    # optional accelerator
    if (isinstance(accelerator_type, str) and accelerator_type
            and isinstance(accelerator_count, int) and accelerator_count > 0):
        machine.update({"accelerator_type": accelerator_type,
                        "accelerator_count": accelerator_count})

    return [{
        "machine_spec": machine,
        "replica_count": 1,
        "container_spec": container,
    }]


def _custom_job(name: str, project: str, location: str, sa: str, specs):
    return CustomTrainingJobOp(
        display_name=name,
        project=project,
        location=location,
        worker_pool_specs=specs,
        service_account=sa
    )


def make_pipeline(run_display_name:str,items: List[str], app_environment:str ="np"):
    @dsl.pipeline(
        name=run_display_name,
        description="Baseline Vertex AI pipeline with fan-out",
    )
    def pipeline(image_uri: str, module_name: str, args_json: str, display_name: str,
                 project_id: str, location: str, service_account: str,
                 machine_type: str = "n1-standard-4",
                 accelerator_type: str = "", accelerator_count: int = 0):
        def _launch(v: Optional[str] = None):
            specs = _worker_pool(image_uri, module_name, args_json, app_environment,
                                 machine_type, v, accelerator_type, accelerator_count)
            job_name = _safe_name(display_name if not v else f"{display_name}-{v}")
            _custom_job(job_name, project_id, location, service_account, specs)

        if items:
            with dsl.ParallelFor(items, name="fanout") as it:
                _launch(v=it)
        else:
            _launch()

    return pipeline


def compile_yaml(func: Any, outfile: str, params: Dict[str, Any]) -> Path:
    # ---- Force UTF-8 for every open() while KFP runs ----
    _orig_open = builtins.open
    _orig_io_open = io.open

    def _utf8_open(file, mode='r', buffering=-1, encoding=None, *a, **kw):
        if 'w' in mode and encoding is None:
            encoding = 'utf-8'
        return _orig_open(file, mode, buffering, encoding, *a, **kw)

    builtins.open = _utf8_open
    io.open = _utf8_open

    try:
        kfp.compiler.Compiler().compile(
            func,
            package_path=outfile,
            pipeline_parameters=params,
        )
    finally:
        builtins.open = _orig_open
        io.open = _orig_io_open

    print(f"  Compiled pipeline YAML ? {outfile}")
    return Path(outfile)


def upload_registry(yaml: Path, project: str, location: str,
                    repo: str, tag: str, desc: str) -> str:
    host = f"https://{location}-kfp.pkg.dev/{project}/{repo}"
    tmpl, _ = RegistryClient(host=host).upload_pipeline(
        file_name=str(yaml), tags=[tag, "latest"],
        extra_headers={"description": desc})
    url = f"{host}/{tmpl}/{tag}"
    print(f"  Uploaded as {url}")
    return url


def run_job(tmpl: str, disp: str, params: Dict[str, str],
            project: str, location: str, pipeline_root: str):
    aiplatform.init(project=project, location=location)
    aiplatform.PipelineJob(
        display_name=disp,
        template_path=tmpl,
        parameter_values=params,
        pipeline_root=pipeline_root,
        enable_caching=False
    ).run()


def _parse():
    p = argparse.ArgumentParser()
    for r in ["image_uri", "module_name", "args_json", "display_name", "project_id",
              "location", "pipeline_root", "service_account"]:
        p.add_argument(f"--{r}", required=True)

    p.add_argument("--parallel_parameter_values", default="[]")
    p.add_argument("--machine_type", default="n1-standard-4")
    p.add_argument("--accelerator_type", default="")
    p.add_argument("--accelerator_count", type=int, default=0)
    p.add_argument("--ensure_bucket", action="store_true")
    p.add_argument("--create_artifact_repo", action="store_true")
    p.add_argument("--artifact_repo_name", default="ml-pipeline-images")
    p.add_argument("--output_path", default="analytical_pipeline.yaml")
    p.add_argument("--upload_registry", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--kfp_repo", default="kfp-repo")
    p.add_argument("--version_tag", default="v1")
    p.add_argument("--description", default="Analytical pipeline template")
    p.add_argument("--run_pipeline", action="store_true")
    p.add_argument("--run_display_name", default="PipelineRun")
    p.add_argument("--parameter_values_json", default="{}")
    p.add_argument("--app_environment",required="True", choices=["np", "prod"], nargs="?", const="np",
                    help="Target environment: np (non-prod), dev, or prod")
    
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    print(f" Using regional API endpoint: {a.location}-aiplatform.googleapis.com")

    if a.ensure_bucket:
        root_bucket = a.pipeline_root.split("/", 3)[0] + "//" + a.pipeline_root.split("/", 3)[2]
        ensure_bucket(root_bucket, a.project_id, a.location)

    if a.create_artifact_repo:
        ensure_artifact_repo(a.artifact_repo_name, a.project_id, a.location)

    # fan-out list handling
    serialised = _parse_list(a.parallel_parameter_values)
    pipeline_def = make_pipeline(a.run_display_name, serialised, a.app_environment)

    bindings = {
        "image_uri": a.image_uri,
        "module_name": a.module_name,
        "args_json": a.args_json,
        "display_name": a.display_name,
        "project_id": a.project_id,
        "location": a.location,
        "service_account": a.service_account,
        "machine_type": a.machine_type,
        "accelerator_type": a.accelerator_type,
        "accelerator_count": a.accelerator_count,
    }

    yml = compile_yaml(pipeline_def, a.output_path, bindings)
    template = (
        upload_registry(yml, a.project_id, a.location, a.kfp_repo,
                        a.version_tag, a.description)
        if a.upload_registry else str(yml)
    )

    if a.run_pipeline:
        base = {
            "image_uri": a.image_uri,
            "module_name": a.module_name,
            "args_json": a.args_json,
            "machine_type": a.machine_type,
        }
        base.update(json.loads(a.parameter_values_json))
        run_job(template, a.run_display_name, base,
                a.project_id, a.location, a.pipeline_root)

    print(" Done.")
# ?---------------------------------------------------------------?
