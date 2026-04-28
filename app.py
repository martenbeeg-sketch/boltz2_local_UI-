import json
import os
import queue
import re
import shutil
import signal
import socketserver
import subprocess
import sys
import threading
import uuid
import hashlib
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from boltz_runner import (
    DEFAULT_BOLTZ_CACHE_DIR,
    DEFAULT_MSA_REPOSITORY_DIR,
    build_boltz_yaml_text,
    collect_metrics,
    find_confidence_files,
    find_structure_path,
    get_cached_msa_paths,
    is_valid_a3m_file,
    parse_entities,
    run_prediction,
)
from visualization import plot_pae, plot_plddt, viewer_html


APP_DIR = Path(__file__).resolve().parent
JOBS_DB_PATH = APP_DIR / "jobs_db.json"
INPUT_REPOSITORY_DIRNAME = "input_repository"
WORKER_SCRIPT_PATH = APP_DIR / "queue_worker.py"
WORKER_PID_PATH = APP_DIR / "queue_worker.pid"
WORKER_LOCK_PATH = APP_DIR / "queue_worker.lock"
EVENT_SERVER_PORT = 8766
EVENT_SERVER_PUBLISH_URL = f"http://127.0.0.1:{EVENT_SERVER_PORT}/publish"

EXAMPLE_PROTEIN = """>THRbeta_human
HKPEPTDEEWELIKTVTEAHVATNAQGSHWKQKRKFLPEDIGQAPIVNAPEGGKVDLEAFSHFTKIITPAITRVVDFAKKLPMFCELPCEDQIILLKGCCMEIMSLRAAVRYDPESETLTLNGEMAVTRGQLKNGGLGVVSDAIFDLGMSLSSFNLDDTEVALLQAVLLMSSDRPGLACVERIEKYQDSFLLAFEHYINYRKHHVTHFWPKLLMKVTDLRMIGACHASRFLHMKVECPTELFPPLFLEVFED"""
EXAMPLE_LIGAND = "OC1=C(I)C=C(OC2=C(I)C=C(C[C@H](N)C(O)=O)C=C2I)C=C1"

ION_PRESETS = {
    "None": "",
    "Na+": "[Na+]",
    "K+": "[K+]",
    "Mg2+": "[Mg+2]",
    "Ca2+": "[Ca+2]",
    "Zn2+": "[Zn+2]",
    "Mn2+": "[Mn+2]",
    "Fe2+": "[Fe+2]",
    "Cu2+": "[Cu+2]",
    "Cl-": "[Cl-]",
}
ENTITY_TYPES = ["protein", "dna", "rna", "ligand", "ion"]
JOB_STATUS_ORDER = ["queued", "running", "completed", "failed"]

EVENT_COMPONENT_JS = """
export default function(component) {
  const { data, setTriggerValue } = component;
  const url = data?.url;
  if (!url) {
    return;
  }
  if (window.__boltzEventBridge && window.__boltzEventBridge.url === url) {
    return;
  }
  if (window.__boltzEventBridge && window.__boltzEventBridge.es) {
    try { window.__boltzEventBridge.es.close(); } catch (e) {}
  }
  const es = new EventSource(url);
  es.onmessage = (ev) => {
    setTriggerValue("event", {
      ts: Date.now(),
      payload: ev.data || ""
    });
  };
  es.onerror = () => {};
  window.__boltzEventBridge = { url, es };
}
"""

_EVENT_CLIENTS: list[queue.Queue[str]] = []
_EVENT_CLIENTS_LOCK = threading.Lock()
_EVENT_SERVER_THREAD: threading.Thread | None = None
_EVENT_SERVER_STARTED = False


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


class _EventHandler(BaseHTTPRequestHandler):
    server_version = "BoltzEventServer/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/publish":
            params = parse_qs(parsed.query)
            raw_event = (params.get("event") or [""])[0]
            if raw_event:
                publish_event(raw_event)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
            return

        if parsed.path != "/events":
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        client_q: queue.Queue[str] = queue.Queue()
        with _EVENT_CLIENTS_LOCK:
            _EVENT_CLIENTS.append(client_q)

        try:
            self.wfile.write(b"event: ready\ndata: connected\n\n")
            self.wfile.flush()
            while True:
                payload = client_q.get()
                line = payload.replace("\n", " ")
                self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                self.wfile.flush()
        except Exception:
            pass
        finally:
            with _EVENT_CLIENTS_LOCK:
                if client_q in _EVENT_CLIENTS:
                    _EVENT_CLIENTS.remove(client_q)

    def log_message(self, fmt: str, *args) -> None:
        return


def publish_event(payload: str) -> None:
    with _EVENT_CLIENTS_LOCK:
        clients = list(_EVENT_CLIENTS)
    for client_q in clients:
        try:
            client_q.put_nowait(payload)
        except Exception:
            pass


def ensure_event_server() -> None:
    global _EVENT_SERVER_THREAD, _EVENT_SERVER_STARTED
    if _EVENT_SERVER_STARTED and _EVENT_SERVER_THREAD and _EVENT_SERVER_THREAD.is_alive():
        return

    def _run_server() -> None:
        httpd = _ThreadingHTTPServer(("127.0.0.1", EVENT_SERVER_PORT), _EventHandler)
        httpd.serve_forever()

    thread = threading.Thread(target=_run_server, name="boltz-event-server", daemon=True)
    thread.start()
    _EVENT_SERVER_THREAD = thread
    _EVENT_SERVER_STARTED = True


def notify_status_event(event: str, job_id: str, status: str, name: str) -> None:
    payload = json.dumps(
        {"event": event, "job_id": job_id, "status": status, "name": name, "at": utc_now()},
        separators=(",", ":"),
    )
    try:
        import urllib.request

        url = f"{EVENT_SERVER_PUBLISH_URL}?{urlencode({'event': payload})}"
        urllib.request.urlopen(url, timeout=1.0).read()
    except Exception:
        pass


def mount_event_listener() -> None:
    if not hasattr(st.components, "v2"):
        return
    listener = st.components.v2.component("boltz_job_event_listener", js=EVENT_COMPONENT_JS)
    listener(
        data={"url": f"http://127.0.0.1:{EVENT_SERVER_PORT}/events"},
        key="boltz_job_event_listener",
        on_event_change=lambda: None,
        height=0,
    )


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def default_entity(entity_type: str = "protein") -> dict:
    return {
        "type": entity_type,
        "copies": 1,
        "input": "",
        "use_affinity": entity_type == "ligand",
        "cyclic": False,
        "ion_choice": "None",
    }


def ensure_state() -> None:
    if "entities" not in st.session_state:
        st.session_state["entities"] = [default_entity("protein")]
    if "job_name" not in st.session_state:
        st.session_state["job_name"] = ""
    if "selected_job_id" not in st.session_state:
        st.session_state["selected_job_id"] = None
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "New Job"
    if "selected_job_ids" not in st.session_state:
        st.session_state["selected_job_ids"] = []
    if "pending_delete_job_ids" not in st.session_state:
        st.session_state["pending_delete_job_ids"] = []


def load_jobs_db() -> list[dict]:
    if not JOBS_DB_PATH.exists():
        return []
    try:
        data = json.loads(JOBS_DB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return data
    return []


def iso_from_epoch(ts: float) -> str:
    return datetime.utcfromtimestamp(ts).isoformat(timespec="seconds") + "Z"


def discover_legacy_jobs(results_dir: str, known_job_dirs: set[str]) -> list[dict]:
    root = Path(results_dir)
    if not root.exists():
        return []
    legacy_jobs: list[dict] = []
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        job_dir = str(child.resolve())
        if job_dir in known_job_dirs:
            continue
        yaml_path = child / "input.yaml"
        if not yaml_path.exists():
            continue
        structure_path = find_structure_path(job_dir)
        confidence_files = find_confidence_files(job_dir)
        metrics = collect_metrics(confidence_files["json_files"])
        updated = iso_from_epoch(child.stat().st_mtime)
        legacy_jobs.append(
            {
                "id": f"legacy:{job_dir}",
                "name": child.name,
                "status": "completed" if structure_path else "failed",
                "created_at": updated,
                "updated_at": updated,
                "entities": [],
                "settings": {},
                "result": {
                    "success": bool(structure_path),
                    "message": "Imported legacy run.",
                    "job_dir": job_dir,
                    "structure_path": structure_path,
                    "structure_format": "cif" if structure_path and structure_path.endswith(".cif") else "pdb",
                    "metrics": metrics,
                    "raw_log": "",
                },
                "error": None if structure_path else "Legacy run missing structure output.",
                "legacy": True,
            }
        )
    return legacy_jobs


def all_jobs_with_legacy(results_dir: str) -> list[dict]:
    db_jobs = load_jobs_db()
    known_dirs = set()
    for job in db_jobs:
        result = job.get("result") or {}
        job_dir = result.get("job_dir")
        if job_dir:
            known_dirs.add(str(Path(job_dir).resolve()))
    for job in db_jobs:
        if job.get("status") in {"queued", "running"}:
            job_name = (job.get("name") or "").strip()
            if job_name:
                known_dirs.add(str((Path(results_dir) / job_name).resolve()))
    legacy_jobs = discover_legacy_jobs(results_dir, known_dirs)
    return db_jobs + legacy_jobs


def compute_yaml_hash(yaml_text: str) -> str:
    return hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()


def get_job_yaml_hash(job: dict) -> str | None:
    existing = job.get("input_hash")
    if existing:
        return existing
    result = job.get("result") or {}
    job_dir = result.get("job_dir")
    if not job_dir:
        return None
    yaml_path = Path(job_dir) / "input.yaml"
    if not yaml_path.exists():
        return None
    try:
        return compute_yaml_hash(yaml_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_duplicate_job(yaml_hash: str, jobs: list[dict]) -> dict | None:
    for job in jobs:
        if get_job_yaml_hash(job) == yaml_hash:
            return job
    return None


def find_duplicate_jobs(yaml_hash: str, jobs: list[dict]) -> list[dict]:
    return [job for job in jobs if get_job_yaml_hash(job) == yaml_hash]


def store_input_yaml_snapshot(cache_dir: str, yaml_hash: str, yaml_text: str) -> None:
    repo = Path(cache_dir) / INPUT_REPOSITORY_DIRNAME
    repo.mkdir(parents=True, exist_ok=True)
    yaml_path = repo / f"{yaml_hash}.yaml"
    if not yaml_path.exists():
        yaml_path.write_text(yaml_text, encoding="utf-8")


def save_jobs_db(jobs: list[dict]) -> None:
    JOBS_DB_PATH.write_text(json.dumps(jobs, indent=2), encoding="utf-8")


def is_worker_running() -> bool:
    if not WORKER_PID_PATH.exists():
        return False
    try:
        pid = int(WORKER_PID_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_worker_lock() -> bool:
    try:
        fd = os.open(str(WORKER_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        if is_worker_running():
            return False
        try:
            WORKER_LOCK_PATH.unlink()
        except Exception:
            return False
        try:
            fd = os.open(str(WORKER_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except Exception:
            return False
    except Exception:
        return False


def start_worker_if_needed() -> bool:
    if is_worker_running():
        return False
    if not _acquire_worker_lock():
        return False
    try:
        subprocess.Popen(
            [sys.executable, str(WORKER_SCRIPT_PATH)],
            cwd=str(APP_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        try:
            WORKER_LOCK_PATH.unlink()
        except Exception:
            pass
        return False
    return True


def backfill_input_hashes(cache_dir: str) -> int:
    jobs = load_jobs_db()
    changed = 0
    for job in jobs:
        if job.get("input_hash"):
            continue
        result = job.get("result") or {}
        job_dir = result.get("job_dir")
        if not job_dir:
            continue
        yaml_path = Path(job_dir) / "input.yaml"
        if not yaml_path.exists():
            continue
        try:
            yaml_text = yaml_path.read_text(encoding="utf-8")
        except Exception:
            continue
        yaml_hash = compute_yaml_hash(yaml_text)
        job["input_hash"] = yaml_hash
        store_input_yaml_snapshot(cache_dir, yaml_hash, yaml_text)
        changed += 1
    if changed:
        save_jobs_db(jobs)
    return changed


def reset_prediction_state() -> None:
    st.session_state["job_name"] = ""
    st.session_state["entities"] = [default_entity("protein")]


def add_entity() -> None:
    st.session_state["entities"].append(default_entity("protein"))


def load_example_entities() -> None:
    example_entities = [
        {
            "type": "protein",
            "copies": 1,
            "input": EXAMPLE_PROTEIN,
            "use_affinity": False,
            "cyclic": False,
            "ion_choice": "None",
        },
        {
            "type": "ligand",
            "copies": 1,
            "input": EXAMPLE_LIGAND,
            "use_affinity": True,
            "cyclic": False,
            "ion_choice": "None",
        },
    ]
    st.session_state["entities"] = example_entities
    for idx, entity in enumerate(example_entities):
        st.session_state[f"entity_type_{idx}"] = entity["type"]
        st.session_state[f"entity_copies_{idx}"] = int(entity["copies"])
        st.session_state[f"entity_input_{idx}"] = entity["input"]
        st.session_state[f"entity_affinity_{idx}"] = bool(entity["use_affinity"])
        st.session_state[f"entity_cyclic_{idx}"] = bool(entity["cyclic"])
        st.session_state[f"entity_ion_choice_{idx}"] = entity["ion_choice"]


def move_entity(index: int, delta: int) -> None:
    new_index = index + delta
    if new_index < 0 or new_index >= len(st.session_state["entities"]):
        return
    entities = st.session_state["entities"]
    entities[index], entities[new_index] = entities[new_index], entities[index]


def remove_entity(index: int) -> None:
    entities = st.session_state["entities"]
    if len(entities) == 1:
        entities[0] = default_entity("protein")
        return
    entities.pop(index)


def sync_entity_inputs() -> None:
    for idx, entity in enumerate(st.session_state["entities"]):
        entity["type"] = st.session_state[f"entity_type_{idx}"]
        entity["copies"] = int(st.session_state[f"entity_copies_{idx}"])
        entity["input"] = st.session_state[f"entity_input_{idx}"]
        entity["cyclic"] = bool(st.session_state.get(f"entity_cyclic_{idx}", False))
        entity["use_affinity"] = bool(st.session_state.get(f"entity_affinity_{idx}", False))
        entity["ion_choice"] = st.session_state.get(f"entity_ion_choice_{idx}", "None")


def effective_entity_input(entity: dict) -> str:
    if entity["type"] == "ion":
        ion_smiles = ION_PRESETS.get(entity.get("ion_choice", "None"), "")
        custom_smiles = (entity.get("input") or "").strip()
        return custom_smiles or ion_smiles
    return entity.get("input", "")


def entities_for_run() -> list[dict]:
    payload: list[dict] = []
    for entity in st.session_state["entities"]:
        payload.append(
            {
                "type": entity["type"],
                "copies": int(entity["copies"]),
                "input": effective_entity_input(entity),
                "use_affinity": bool(entity.get("use_affinity", False) and entity["type"] == "ligand"),
                "cyclic": bool(entity.get("cyclic", False)),
            }
        )
    return payload


def queue_job(job_name: str, entities: list[dict], settings: dict, input_hash: str | None = None) -> str:
    jobs = load_jobs_db()
    job_id = str(uuid.uuid4())
    now = utc_now()
    timestamp_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", job_name).strip("_") or "job"
    run_name = safe_name[:30]
    display_name = f"{timestamp_label}_{run_name}"
    jobs.append(
        {
            "id": job_id,
            "name": display_name,
            "run_name": run_name,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "entities": entities,
            "settings": settings,
            "input_hash": input_hash,
            "result": None,
            "error": None,
        }
    )
    save_jobs_db(jobs)
    notify_status_event("queued", job_id, "queued", display_name)
    return job_id


def cleanup_yaml_snapshot_if_unreferenced(yaml_hash: str, cache_dir: str, results_dir: str) -> bool:
    if not yaml_hash:
        return False
    jobs = all_jobs_with_legacy(results_dir)
    still_used = any(get_job_yaml_hash(job) == yaml_hash for job in jobs)
    if still_used:
        return False
    yaml_path = Path(cache_dir) / INPUT_REPOSITORY_DIRNAME / f"{yaml_hash}.yaml"
    if yaml_path.exists():
        yaml_path.unlink()
        return True
    return False


def remove_job_entry(job_id: str, cache_dir: str, results_dir: str) -> tuple[bool, str]:
    removed_hash: str | None = None
    # Legacy-only entry: remove the folder directly.
    if job_id.startswith("legacy:"):
        job_dir = job_id.split("legacy:", 1)[1]
        job_path = Path(job_dir)
        if not job_path.exists():
            return False, "Legacy job folder does not exist anymore."
        yaml_path = job_path / "input.yaml"
        if yaml_path.exists():
            try:
                removed_hash = compute_yaml_hash(yaml_path.read_text(encoding="utf-8"))
            except Exception:
                removed_hash = None
        shutil.rmtree(job_path, ignore_errors=True)
        zip_path = Path(f"{job_dir}.zip")
        if zip_path.exists():
            zip_path.unlink()
        if removed_hash:
            cleanup_yaml_snapshot_if_unreferenced(removed_hash, cache_dir, results_dir)
        return True, f"Removed legacy job folder: {job_path.name}"

    jobs = load_jobs_db()
    idx = next((i for i, j in enumerate(jobs) if j["id"] == job_id), None)
    if idx is None:
        return False, "Job not found."

    job = jobs.pop(idx)
    removed_name = job.get("name", job_id)
    removed_hash = job.get("input_hash") or get_job_yaml_hash(job)
    result = job.get("result") or {}
    job_dir = result.get("job_dir")
    save_jobs_db(jobs)

    if job_dir:
        job_path = Path(job_dir)
        if job_path.exists():
            shutil.rmtree(job_path, ignore_errors=True)
        zip_path = Path(f"{job_dir}.zip")
        if zip_path.exists():
            zip_path.unlink()

    removed_yaml_snapshot = False
    if removed_hash:
        removed_yaml_snapshot = cleanup_yaml_snapshot_if_unreferenced(removed_hash, cache_dir, results_dir)

    suffix = " (yaml snapshot cleaned)" if removed_yaml_snapshot else ""
    return True, f"Removed job: {removed_name}{suffix}"


def remove_selected_jobs(job_ids: list[str], cache_dir: str, results_dir: str) -> tuple[int, list[str]]:
    removed = 0
    errors: list[str] = []
    for job_id in job_ids:
        ok, msg = remove_job_entry(job_id, cache_dir, results_dir)
        if ok:
            removed += 1
        else:
            errors.append(msg)
    return removed, errors


def process_next_job() -> tuple[bool, str]:
    jobs = load_jobs_db()
    queued_index = next((idx for idx, j in enumerate(jobs) if j["status"] == "queued"), None)
    if queued_index is None:
        return False, "No queued jobs."

    job = jobs[queued_index]
    job["status"] = "running"
    job["updated_at"] = utc_now()
    save_jobs_db(jobs)
    notify_status_event("running", job["id"], "running", job.get("name", job["id"]))

    settings = job["settings"]
    run_name = (job.get("run_name") or job.get("name") or "job").strip()
    result = run_prediction(
        "",
        "",
        entities=job["entities"],
        job_name=run_name,
        results_dir=settings["results_dir"],
        cache_dir=settings["cache_dir"],
        msa_repository_dir=settings["msa_repository_dir"],
        docker_image=settings["docker_image"],
        docker_args=settings["docker_args"],
        use_msa_repository=settings["use_msa_repository"],
        use_potentials=settings["use_potentials"],
        enable_affinity=settings["enable_affinity"],
        sampling_steps=int(settings["sampling_steps"]),
        recycling_steps=int(settings["recycling_steps"]),
        diffusion_samples=int(settings["diffusion_samples"]),
        sampling_steps_affinity=int(settings["sampling_steps_affinity"]),
        diffusion_samples_affinity=int(settings["diffusion_samples_affinity"]),
        affinity_mw_correction=bool(settings["affinity_mw_correction"]),
        num_copies=1,
        cyclic=False,
        use_msa_server=settings["use_msa_server"],
        gpu_device=settings["gpu_device"],
    )

    jobs = load_jobs_db()
    idx = next((i for i, j in enumerate(jobs) if j["id"] == job["id"]), None)
    if idx is None:
        return False, "Job disappeared from queue database."

    jobs[idx]["updated_at"] = utc_now()
    jobs[idx]["status"] = "completed" if result.success else "failed"
    jobs[idx]["result"] = {
        "success": bool(result.success),
        "message": result.message,
        "job_dir": result.job_dir,
        "structure_path": result.structure_path,
        "structure_format": result.structure_format,
        "metrics": result.metrics,
        "raw_log": result.raw_log,
    }
    jobs[idx]["error"] = None if result.success else result.message
    save_jobs_db(jobs)
    notify_status_event(
        jobs[idx]["status"],
        jobs[idx]["id"],
        jobs[idx]["status"],
        jobs[idx].get("name", jobs[idx]["id"]),
    )
    return True, f"Processed job: {job['name']} ({jobs[idx]['status']})"


def status_emoji(status: str) -> str:
    return {
        "queued": "⏳",
        "running": "🏃",
        "completed": "✅",
        "failed": "❌",
    }.get(status, "•")


def select_all_filtered_jobs(job_ids: list[str]) -> None:
    selected = set(st.session_state.get("selected_job_ids", []))
    for jid in job_ids:
        selected.add(jid)
        st.session_state[f"landing_select_{jid}"] = True
    st.session_state["selected_job_ids"] = list(selected)


def deselect_all_filtered_jobs(job_ids: list[str]) -> None:
    selected = set(st.session_state.get("selected_job_ids", []))
    for jid in job_ids:
        selected.discard(jid)
        st.session_state[f"landing_select_{jid}"] = False
    st.session_state["selected_job_ids"] = list(selected)


def render_result_view(job_record: dict) -> None:
    result = job_record.get("result") or {}
    st.subheader(job_record["name"])
    st.caption(f"Status: {job_record['status']} • Updated: {job_record.get('updated_at', '-')}")
    if not result:
        if job_record.get("error"):
            st.error(job_record["error"])
        else:
            st.info("No result data yet.")
        return
    if not result.get("success", False):
        st.error(result.get("message", "Job failed."))
        if result.get("raw_log"):
            st.text_area("Logs", result["raw_log"], height=320)
        return

    st.success(result.get("message", "Completed"))
    job_dir = result.get("job_dir")
    if job_dir:
        st.code(f"Result folder: {job_dir}")
    metrics = result.get("metrics") or {}
    if metrics:
        display_metrics = dict(metrics)
        affinity_value = display_metrics.get("affinity")
        if isinstance(affinity_value, (int, float)):
            display_metrics["ic50_uM"] = float(10**affinity_value)
            display_metrics["pIC50"] = float(6 - affinity_value)
            display_metrics["affinity_kcalmol_eq"] = float((6 - affinity_value) * 1.364)
        metric_columns = st.columns(min(4, max(1, len(display_metrics))))
        for index, (key, value) in enumerate(display_metrics.items()):
            if key == "binding_probability":
                display = f"{value:.2%}"
            elif key == "ic50_uM":
                display = f"{value:.3g} µM"
            elif isinstance(value, (int, float)):
                display = f"{value:.2f}"
            else:
                display = str(value)
            metric_columns[index % len(metric_columns)].metric(key, display)

    structure_path = result.get("structure_path")
    if structure_path and Path(structure_path).exists():
        structure_text = Path(structure_path).read_text(encoding="utf-8")
        components.html(viewer_html(structure_text, result.get("structure_format") or "cif"), height=520)
        structure_file = Path(structure_path)
        st.download_button(
            "Download structure",
            data=structure_file.read_bytes(),
            file_name=structure_file.name,
            mime="chemical/x-cif" if structure_file.suffix == ".cif" else "chemical/x-pdb",
            use_container_width=True,
        )
    if job_dir and Path(job_dir).exists():
        archive_path = Path(shutil.make_archive(job_dir, "zip", root_dir=job_dir))
        st.download_button(
            "Download full results",
            data=archive_path.read_bytes(),
            file_name=archive_path.name,
            mime="application/zip",
            use_container_width=True,
        )
    st.text_area("Logs", result.get("raw_log", ""), height=320)


ensure_state()
st.set_page_config(page_title="Boltz-2 Local", page_icon="🧬", layout="wide")
ensure_event_server()
mount_event_listener()
st.title("Boltz-2 Local")
st.caption("Streamlit host UI that runs Boltz-2 through a Docker container.")

with st.sidebar:
    st.subheader("Runtime")
    gpu_device = st.selectbox("GPU device", options=["0", "1", "all"], index=0)
    with st.expander("Settings", expanded=False):
        docker_image = st.text_input("Docker image", value=st.session_state.get("docker_image", "ovoex-boltz2"), key="docker_image")
        cache_dir = st.text_input("Cache directory", value=st.session_state.get("cache_dir", DEFAULT_BOLTZ_CACHE_DIR), key="cache_dir")
        msa_repository_dir = st.text_input(
            "MSA repository directory",
            value=st.session_state.get("msa_repository_dir", DEFAULT_MSA_REPOSITORY_DIR),
            key="msa_repository_dir",
        )
        results_dir = st.text_input(
            "Results directory",
            value=st.session_state.get("results_dir", str((Path(__file__).resolve().parent / "results"))),
            key="results_dir",
        )
        docker_args = st.text_input("Docker extra args", value=st.session_state.get("docker_args", "--ipc=host --shm-size=48G"), key="docker_args")
    st.code(f"BOLTZ_DOCKER_IMAGE={docker_image}\nBOLTZ_CACHE_DIR={cache_dir}", language="bash")

    with st.expander("Boltz settings", expanded=False):
        use_msa_server = st.checkbox("Use MSA", value=True)
        use_msa_repository = st.checkbox("Use local MSA repository", value=True)
        use_potentials = st.checkbox("Respect physics (use potentials)", value=True)
        enable_affinity = st.checkbox("Enable affinity for ligand runs", value=True)
        sampling_steps = st.number_input("Sampling steps", min_value=10, max_value=400, value=200, step=10)
        recycling_steps = st.number_input("Recycling steps", min_value=1, max_value=12, value=3, step=1)
        diffusion_samples = st.number_input("Diffusion samples", min_value=1, max_value=16, value=1, step=1)
        sampling_steps_affinity = st.number_input("Affinity sampling steps", min_value=10, max_value=400, value=200, step=10)
        diffusion_samples_affinity = st.number_input("Affinity diffusion samples", min_value=1, max_value=16, value=5, step=1)
        affinity_mw_correction = st.checkbox("Affinity molecular-weight correction", value=False)

    if st.button("Stop application", type="secondary", use_container_width=True):
        st.warning("Stopping Streamlit. This browser tab will disconnect and the port will be freed.")
        threading.Timer(0.75, lambda: os.kill(os.getpid(), signal.SIGINT)).start()

st.markdown(
    """
<style>
div.block-container {{
    max-width: min(2200px, 96vw);
    padding-top: 1.2rem;
    padding-left: 1.4rem;
    padding-right: 1.4rem;
}}
</style>
""",
    unsafe_allow_html=True,
)

runtime_settings = {
    "gpu_device": gpu_device,
    "docker_image": docker_image,
    "cache_dir": cache_dir,
    "msa_repository_dir": msa_repository_dir,
    "results_dir": results_dir,
    "docker_args": docker_args,
    "use_msa_server": use_msa_server,
    "use_msa_repository": use_msa_repository,
    "use_potentials": use_potentials,
    "enable_affinity": enable_affinity,
    "sampling_steps": int(sampling_steps),
    "recycling_steps": int(recycling_steps),
    "diffusion_samples": int(diffusion_samples),
    "sampling_steps_affinity": int(sampling_steps_affinity),
    "diffusion_samples_affinity": int(diffusion_samples_affinity),
    "affinity_mw_correction": bool(affinity_mw_correction),
}

if "input_hash_backfill_done" not in st.session_state:
    updated_hash_count = backfill_input_hashes(runtime_settings["cache_dir"])
    st.session_state["input_hash_backfill_done"] = True
    st.session_state["input_hash_backfill_count"] = updated_hash_count

if st.session_state["nav_page"] != "Job Details":
    if "jobs_flash" in st.session_state:
        flash_msg = st.session_state.pop("jobs_flash")
        if flash_msg.startswith("Delete failed:"):
            st.error(flash_msg)
        else:
            st.success(flash_msg)

    if st.session_state.get("input_hash_backfill_count", 0):
        st.info(
            f"Backfilled YAML hashes for {st.session_state['input_hash_backfill_count']} existing job(s)."
        )

    st.text_input("Job name", key="job_name", placeholder="example: insulin_test_01")
    st.subheader("Entities")
    for idx, entity in enumerate(st.session_state["entities"]):
        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.selectbox(
                    "Type",
                    ENTITY_TYPES,
                    index=ENTITY_TYPES.index(entity["type"]) if entity["type"] in ENTITY_TYPES else 0,
                    key=f"entity_type_{idx}",
                )
            with c2:
                st.number_input("Copies", min_value=1, max_value=8, value=int(entity["copies"]), key=f"entity_copies_{idx}")
            with c3:
                u, d, x = st.columns(3)
                with u:
                    st.button("↑", key=f"entity_up_{idx}", on_click=move_entity, args=(idx, -1))
                with d:
                    st.button("↓", key=f"entity_down_{idx}", on_click=move_entity, args=(idx, 1))
                with x:
                    st.button("✕", key=f"entity_del_{idx}", on_click=remove_entity, args=(idx,))

            current_type = st.session_state.get(f"entity_type_{idx}", entity["type"])
            if current_type == "ion":
                st.selectbox("Ion preset", list(ION_PRESETS.keys()), key=f"entity_ion_choice_{idx}")
                st.text_input("Custom ion SMILES (optional)", value=entity.get("input", ""), key=f"entity_input_{idx}")
                st.session_state[f"entity_affinity_{idx}"] = False
            elif current_type == "ligand":
                st.text_input("Ligand SMILES", value=entity.get("input", ""), key=f"entity_input_{idx}")
                st.checkbox("Affinity for this entity", value=bool(entity.get("use_affinity", True)), key=f"entity_affinity_{idx}")
            else:
                st.text_area("Sequence (FASTA or raw)", value=entity.get("input", ""), key=f"entity_input_{idx}", height=120)
                if current_type == "protein":
                    st.checkbox("Cyclic protein", value=bool(entity.get("cyclic", False)), key=f"entity_cyclic_{idx}")

    b1, b2 = st.columns(2)
    with b1:
        st.button("Add entity", on_click=add_entity, use_container_width=True)
    with b2:
        st.button("Load example", on_click=load_example_entities, use_container_width=True)

    sync_entity_inputs()
    effective_entities = entities_for_run()
    normalized_entities, first_protein_sequence, parse_message = parse_entities(effective_entities)
    ok_entities = bool(normalized_entities)
    cached_msa_host = ""
    cached_msa_container = ""
    yaml_msa_path = None
    msa_cache_invalid_reason = ""

    if ok_entities and first_protein_sequence:
        cached_msa_host, cached_msa_container = get_cached_msa_paths(
            sequence=first_protein_sequence, msa_repository_dir=runtime_settings["msa_repository_dir"]
        )
        if runtime_settings["use_msa_repository"] and os.path.exists(cached_msa_host):
            valid_msa, validation_reason = is_valid_a3m_file(cached_msa_host)
            if valid_msa:
                yaml_msa_path = cached_msa_container
            else:
                msa_cache_invalid_reason = validation_reason

    if runtime_settings["use_msa_repository"]:
        if not ok_entities:
            st.caption("MSA cache status: add at least one valid protein entity.")
        elif yaml_msa_path:
            st.success(f"MSA available in repository: {cached_msa_host}")
        elif msa_cache_invalid_reason:
            st.error(
                "MSA cache file exists but is invalid and will be ignored: "
                f"{cached_msa_host} ({msa_cache_invalid_reason})"
            )
        else:
            st.warning("MSA not cached yet for this protein sequence. Server will be used once and then cached.")

    with st.popover("Preview YAML", use_container_width=True):
        if not ok_entities:
            st.error(parse_message)
        else:
            yaml_preview = build_boltz_yaml_text(
                sequence=first_protein_sequence or "",
                entities=normalized_entities,
                msa_path=yaml_msa_path,
                enable_affinity=runtime_settings["enable_affinity"],
            )
            st.code(yaml_preview, language="yaml")

    # Real-time duplicate detection (same behavior style as MSA presence check).
    candidate_hash_live = None
    if ok_entities:
        candidate_yaml_live = build_boltz_yaml_text(
            sequence=first_protein_sequence or "",
            entities=normalized_entities,
            msa_path=None,
            enable_affinity=runtime_settings["enable_affinity"],
        )
        candidate_hash_live = compute_yaml_hash(candidate_yaml_live)
        duplicate_live = find_duplicate_jobs(
            candidate_hash_live,
            all_jobs_with_legacy(runtime_settings["results_dir"]),
        )
        if duplicate_live:
            st.warning(f"Duplicate input detected in {len(duplicate_live)} job(s).")
            with st.expander("Show matching jobs", expanded=False):
                for dup in duplicate_live:
                    st.caption(
                        f"- {dup['name']} | {dup.get('status', 'unknown')} | {dup.get('updated_at', '-')}"
                    )
        else:
            st.caption("No duplicate input found for current YAML.")

    if st.button("Run job", type="primary", use_container_width=True):
        if not st.session_state["job_name"].strip():
            st.error("Job name is required.")
        elif not ok_entities:
            st.error(parse_message)
        else:
            candidate_yaml = build_boltz_yaml_text(
                sequence=first_protein_sequence or "",
                entities=normalized_entities,
                msa_path=None,
                enable_affinity=runtime_settings["enable_affinity"],
            )
            candidate_hash = candidate_hash_live or compute_yaml_hash(candidate_yaml)
            all_known_jobs = all_jobs_with_legacy(runtime_settings["results_dir"])
            duplicates = find_duplicate_jobs(candidate_hash, all_known_jobs)
            if duplicates:
                st.warning(
                    f"Identical input already exists in {len(duplicates)} job(s). "
                    "Submitting new job anyway."
                )
                with st.expander("Show matching jobs", expanded=False):
                    for dup in duplicates:
                        st.caption(
                            f"- {dup['name']} | {dup.get('status', 'unknown')} | {dup.get('updated_at', '-')}"
                        )
            store_input_yaml_snapshot(runtime_settings["cache_dir"], candidate_hash, candidate_yaml)
            job_id = queue_job(
                st.session_state["job_name"].strip(),
                normalized_entities,
                runtime_settings,
                input_hash=candidate_hash,
            )
            spawned_worker = start_worker_if_needed()
            if spawned_worker:
                st.success(f"Job queued: {job_id}. Background worker started.")
            else:
                st.success(f"Job queued: {job_id}.")
            st.info(f"Input hash: {candidate_hash}")

    st.subheader("Queue Summary")
    jobs = all_jobs_with_legacy(runtime_settings["results_dir"])
    counts = {status: 0 for status in JOB_STATUS_ORDER}
    for job in jobs:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    c = st.columns(4)
    c[0].metric("Queued", counts.get("queued", 0))
    c[1].metric("Running", counts.get("running", 0))
    c[2].metric("Completed", counts.get("completed", 0))
    c[3].metric("Failed", counts.get("failed", 0))
    st.caption(f"Queue worker: {'running' if is_worker_running() else 'idle'}")

    st.subheader("Recent Jobs")
    inline_query = st.text_input("Search jobs", value="", key="landing_jobs_query")
    inline_status = st.multiselect(
        "Status",
        JOB_STATUS_ORDER,
        default=JOB_STATUS_ORDER,
        key="landing_jobs_status_filter",
    )
    filtered_inline = [
        j
        for j in jobs
        if j["status"] in inline_status
        and (not inline_query.strip() or inline_query.lower().strip() in j["name"].lower())
    ]
    filtered_inline = sorted(filtered_inline, key=lambda j: j.get("name", ""), reverse=True)[:12]
    filtered_ids = [j["id"] for j in filtered_inline]
    actions_c1, actions_c2 = st.columns(2)
    with actions_c1:
        st.button(
            "Select all filtered",
            use_container_width=True,
            on_click=select_all_filtered_jobs,
            args=(filtered_ids,),
            disabled=not filtered_ids,
        )
    with actions_c2:
        st.button(
            "Deselect all filtered",
            use_container_width=True,
            on_click=deselect_all_filtered_jobs,
            args=(filtered_ids,),
            disabled=not filtered_ids,
        )


    if not filtered_inline:
        st.caption("No jobs found.")
    selected_now = set(st.session_state.get("selected_job_ids", []))
    for job in filtered_inline:
        cols = st.columns([0.7, 4, 2, 2, 1])
        is_selected = job["id"] in selected_now
        cols[0].checkbox(
            "Select",
            value=is_selected,
            key=f"landing_select_{job['id']}",
            label_visibility="collapsed",
        )
        if st.session_state.get(f"landing_select_{job['id']}", False):
            selected_now.add(job["id"])
        else:
            selected_now.discard(job["id"])
        cols[1].write(f"**{job['name']}**")
        cols[2].write(job["status"])
        cols[3].write(job.get("updated_at", "-"))
        if cols[4].button("Open", key=f"landing_open_{job['id']}"):
            st.session_state["selected_job_id"] = job["id"]
            st.session_state["nav_page"] = "Job Details"
            st.rerun()
    st.session_state["selected_job_ids"] = list(selected_now)

    st.divider()
    delete_selected = st.button(
        "Delete selected jobs",
        type="secondary",
        use_container_width=True,
        disabled=not st.session_state.get("selected_job_ids", []),
    )
    if delete_selected:
        st.session_state["pending_delete_job_ids"] = list(st.session_state.get("selected_job_ids", []))
        st.rerun()

    pending_ids = list(st.session_state.get("pending_delete_job_ids", []))
    if pending_ids:
        pending_jobs = [j for j in jobs if j["id"] in set(pending_ids)]
        st.warning("Delete confirmation required.")
        st.write("Selected jobs to delete:")
        for pending_job in pending_jobs:
            st.write(f"- {pending_job['name']} ({pending_job['status']})")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Confirm delete", type="primary", use_container_width=True):
                removed_count, errors = remove_selected_jobs(
                    pending_ids,
                    runtime_settings["cache_dir"],
                    runtime_settings["results_dir"],
                )
                st.session_state["selected_job_ids"] = [
                    jid for jid in st.session_state.get("selected_job_ids", []) if jid not in set(pending_ids)
                ]
                st.session_state["pending_delete_job_ids"] = []
                if removed_count:
                    st.session_state["jobs_flash"] = f"Removed {removed_count} selected job(s)."
                if errors:
                    st.session_state["jobs_flash"] = (
                        st.session_state.get("jobs_flash", "")
                        + (" " if st.session_state.get("jobs_flash") else "")
                        + f" Delete errors: {'; '.join(errors)}"
                    )
                st.rerun()
        with c2:
            if st.button("Cancel delete", use_container_width=True):
                st.session_state["pending_delete_job_ids"] = []
                st.rerun()

else:
    jobs = all_jobs_with_legacy(runtime_settings["results_dir"])
    selected_id = st.session_state.get("selected_job_id")
    job = next((j for j in jobs if j["id"] == selected_id), None)
    top_cols = st.columns([1, 4])
    if top_cols[0].button("Back to New Job"):
        st.session_state["nav_page"] = "New Job"
        st.rerun()
    if job is None:
        st.warning("No job selected.")
    else:
        render_result_view(job)
