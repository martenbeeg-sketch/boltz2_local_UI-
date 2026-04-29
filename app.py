import json
import io
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
from copy import deepcopy
import zipfile

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
EXAMPLE_RBDWT = "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"

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
METRIC_HELP = {
    "complex_plddt": "Local structure confidence for the full complex (0-1). Higher is better.",
    "complex_pde": "Local distance error proxy in Angstrom. Lower is better.",
    "ptm": "Global fold/topology confidence (0-1). Higher is better.",
    "confidence_score": "Overall confidence score used for ranking. Higher is better.",
    "iptm": "Interface confidence across interacting chains (0-1). Higher is better.",
    "ligand_iptm": "Interface confidence for protein-ligand contacts. Higher is better.",
    "protein_iptm": "Interface confidence for protein-protein contacts. Higher is better.",
    "complex_iplddt": "Interface-weighted local confidence. Higher is better.",
    "complex_ipde": "Interface-weighted distance error in Angstrom. Lower is better.",
    "affinity": "Model affinity output: log10(IC50 in uM). Lower means stronger predicted binding.",
    "ic50_uM": "IC50 estimate derived from affinity, in micromolar (uM). Lower is better.",
    "binding_probability": "Predicted probability that ligand is a binder (0-1). Higher is better.",
}

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
    if "selected_batch_id" not in st.session_state:
        st.session_state["selected_batch_id"] = None
    if "last_batch_mode" not in st.session_state:
        st.session_state["last_batch_mode"] = "none"
    if "pending_delete_batch_id" not in st.session_state:
        st.session_state["pending_delete_batch_id"] = None
    if "selected_batch_ids" not in st.session_state:
        st.session_state["selected_batch_ids"] = []
    if "pending_delete_batch_ids" not in st.session_state:
        st.session_state["pending_delete_batch_ids"] = []
    if "batch_page" not in st.session_state:
        st.session_state["batch_page"] = 1
    if "jobs_page" not in st.session_state:
        st.session_state["jobs_page"] = 1


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


def parse_batch_items(file_text: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for raw_line in file_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            left, right = line.split(",", 1)
        elif "\t" in line:
            left, right = line.split("\t", 1)
        else:
            continue
        item_id = left.strip()
        item_value = right.strip()
        if not item_id or not item_value:
            continue
        if item_id.lower() in {"id", "name"}:
            continue
        items.append((item_id, item_value))
    return items


def batch_groups(jobs: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for job in jobs:
        batch_id = job.get("batch_id")
        if not batch_id:
            continue
        if batch_id not in grouped:
            grouped[batch_id] = {
                "batch_id": batch_id,
                "batch_name": job.get("batch_name", batch_id),
                "created_at": job.get("created_at", ""),
                "updated_at": job.get("updated_at", ""),
                "jobs": [],
            }
        grouped[batch_id]["jobs"].append(job)
        if job.get("updated_at", "") > grouped[batch_id]["updated_at"]:
            grouped[batch_id]["updated_at"] = job.get("updated_at", "")
    result = []
    for group in grouped.values():
        counts = {status: 0 for status in JOB_STATUS_ORDER}
        for job in group["jobs"]:
            counts[job.get("status", "queued")] = counts.get(job.get("status", "queued"), 0) + 1
        group["counts"] = counts
        group["total"] = len(group["jobs"])
        if counts.get("running", 0) > 0:
            group["status"] = "running"
        elif counts.get("queued", 0) > 0:
            group["status"] = "queued"
        elif counts.get("failed", 0) > 0:
            group["status"] = "failed"
        else:
            group["status"] = "completed"
        result.append(group)
    return sorted(result, key=lambda g: g.get("updated_at", ""), reverse=True)


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


def on_batch_mode_change() -> None:
    # Reset entity editor to a blank default when switching batch mode.
    st.session_state["entities"] = [default_entity("protein")]
    st.session_state["batch_text_input"] = ""
    for key in list(st.session_state.keys()):
        if key.startswith("entity_"):
            del st.session_state[key]
    st.session_state["entity_type_0"] = "protein"
    st.session_state["entity_copies_0"] = 1
    st.session_state["entity_input_0"] = ""
    st.session_state["entity_affinity_0"] = False
    st.session_state["entity_cyclic_0"] = False
    st.session_state["entity_ion_choice_0"] = "None"


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


def load_batch_example(batch_mode: str) -> None:
    # Batch scaffold: fixed target sequence only (no default ligand entity).
    batch_scaffold = [
        {
            "type": "protein",
            "copies": 1,
            "input": EXAMPLE_PROTEIN,
            "use_affinity": False,
            "cyclic": False,
            "ion_choice": "None",
        }
    ]
    st.session_state["entities"] = batch_scaffold
    st.session_state["entity_type_0"] = "protein"
    st.session_state["entity_copies_0"] = 1
    st.session_state["entity_input_0"] = EXAMPLE_RBDWT if batch_mode == "batch_sequences" else EXAMPLE_PROTEIN
    st.session_state["entity_affinity_0"] = False
    st.session_state["entity_cyclic_0"] = False
    st.session_state["entity_ion_choice_0"] = "None"
    if batch_mode == "batch_sequences":
        st.session_state["batch_text_input"] = (
            "LCB1,DKEWILQKIYEIMRLLDELGHAEASMRVSDLIYEFMKKGDERLLEEAERLLEEVER\n"
            "LCB3,NDDELHMLMTDLVYEALHFAKDEEIKKRVFQLFELADKAYKNNDRQKLEKVVEELKELLERLLS\n"
            "LCB8,PIIELLREAKEKNDEFAISDALYLVNELLQRTGDPRLEEVLYLIWRALKEKDPRLLDRAIELFER"
        )
    else:
        st.session_state["batch_text_input"] = (
            "LIG_001,OC1=C(I)C=C(OC2=C(I)C=C(C[C@H](N)C(O)=O)C=C2I)C=C1\n"
            "LIG_002,CC(=O)OC1=CC=CC=C1C(=O)O\n"
            "LIG_003,CN1CCC[C@H]1C2=CN=CC=C2"
        )


def load_context_example(batch_mode: str) -> None:
    if batch_mode == "batch_ligands":
        load_batch_example("batch_ligands")
        return
    if batch_mode == "batch_sequences":
        load_batch_example("batch_sequences")
        return
    load_example_entities()


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


def queue_job(
    job_name: str,
    entities: list[dict],
    settings: dict,
    input_hash: str | None = None,
    *,
    batch_id: str | None = None,
    batch_name: str | None = None,
    batch_item_id: str | None = None,
    batch_role: str | None = None,
) -> str:
    jobs = load_jobs_db()
    job_id = str(uuid.uuid4())
    now = utc_now()
    timestamp_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", job_name).strip("_") or "job"
    run_name = safe_name[:30]
    display_name = f"{timestamp_label}_{run_name}"
    # Use the displayed timestamped job name for execution as well, so
    # result folder naming stays in sync with table naming.
    run_name = display_name
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
            "batch_id": batch_id,
            "batch_name": batch_name,
            "batch_item_id": batch_item_id,
            "batch_role": batch_role,
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


def remove_batch_jobs(batch_id: str, cache_dir: str, results_dir: str) -> tuple[int, list[str]]:
    jobs = load_jobs_db()
    batch_job_ids = [j["id"] for j in jobs if j.get("batch_id") == batch_id]
    if not batch_job_ids:
        return 0, [f"No persisted jobs found for batch {batch_id}."]
    return remove_selected_jobs(batch_job_ids, cache_dir, results_dir)


def remove_selected_batches(batch_ids: list[str], cache_dir: str, results_dir: str) -> tuple[int, list[str]]:
    total_removed = 0
    all_errors: list[str] = []
    for batch_id in batch_ids:
        removed, errors = remove_batch_jobs(batch_id, cache_dir, results_dir)
        total_removed += removed
        all_errors.extend(errors)
    return total_removed, all_errors


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


def select_all_filtered_batches(batch_ids: list[str]) -> None:
    selected = set(st.session_state.get("selected_batch_ids", []))
    for batch_id in batch_ids:
        selected.add(batch_id)
        st.session_state[f"batch_select_{batch_id}"] = True
    st.session_state["selected_batch_ids"] = list(selected)


def deselect_all_filtered_batches(batch_ids: list[str]) -> None:
    selected = set(st.session_state.get("selected_batch_ids", []))
    for batch_id in batch_ids:
        selected.discard(batch_id)
        st.session_state[f"batch_select_{batch_id}"] = False
    st.session_state["selected_batch_ids"] = list(selected)


def merged_metrics_for_job(job: dict) -> dict:
    result = job.get("result") or {}
    metrics = dict(result.get("metrics") or {})
    job_dir = result.get("job_dir")
    if job_dir and Path(job_dir).exists():
        try:
            files = find_confidence_files(job_dir)
            live_metrics = collect_metrics(files.get("json_files") or [])
            if isinstance(live_metrics, dict):
                metrics.update(live_metrics)
        except Exception:
            pass
    return metrics


def build_results_zip_bytes(job_dir: str) -> bytes:
    root = Path(job_dir)
    if not root.exists():
        return b""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(root)))
    return buf.getvalue()


def build_ligand_batch_summary_rows(members: list[dict]) -> list[dict]:
    rows: list[dict] = []
    metric_keys: set[str] = set()
    for job in members:
        metrics = merged_metrics_for_job(job)
        metric_keys.update(metrics.keys())

    for job in members:
        metrics = merged_metrics_for_job(job)
        affinity = metrics.get("affinity")
        binding_probability = metrics.get("binding_probability")
        ptm = metrics.get("ptm")
        row = {
            "item_id": job.get("batch_item_id") or job.get("name", ""),
            "job_name": job.get("name", ""),
            "status": job.get("status", ""),
            "updated_at": job.get("updated_at", ""),
            "ptm": ptm if isinstance(ptm, (int, float)) else None,
        }
        for mk in metric_keys:
            value = metrics.get(mk)
            row[mk] = value if isinstance(value, (int, float, str)) or value is None else str(value)
        if isinstance(binding_probability, (int, float)):
            row["binding_probability"] = float(binding_probability)
        if isinstance(affinity, (int, float)):
            row["affinity"] = float(affinity)
            row["ic50_uM"] = float(10**affinity)
            row["pIC50"] = float(6 - affinity)
        rows.append(row)

    def sort_key(r: dict):
        # Lower affinity value indicates stronger binder for this model.
        if isinstance(r.get("affinity"), (int, float)):
            return (0, float(r["affinity"]))
        return (1, r.get("job_name", ""))

    rows.sort(key=sort_key)
    return rows


def get_cached_ligand_batch_summary(batch_id: str, members: list[dict]) -> list[dict]:
    if "batch_summary_cache" not in st.session_state:
        st.session_state["batch_summary_cache"] = {}
    cache: dict = st.session_state["batch_summary_cache"]
    signature = tuple(sorted((j.get("id", ""), j.get("status", ""), j.get("updated_at", "")) for j in members))
    entry = cache.get(batch_id)
    if entry and entry.get("signature") == signature:
        return entry.get("rows", [])
    rows = build_ligand_batch_summary_rows(members)
    cache[batch_id] = {"signature": signature, "rows": rows}
    st.session_state["batch_summary_cache"] = cache
    return rows


def render_ligand_batch_summary(batch_id: str, members: list[dict]) -> list[dict]:
    if not members:
        return []
    batch_role = members[0].get("batch_role")
    rows = get_cached_ligand_batch_summary(batch_id, members)
    if not rows:
        return []
    title = "Batch Summary (Ligands)" if batch_role == "ligand" else "Batch Summary (Sequences)"
    st.markdown(f"**{title}**")
    st.caption("Use search/filter/sort to organize the batch table.")

    controls = st.columns([2, 2, 2, 1])
    search_query = controls[0].text_input(
        "Search rows",
        value="",
        key=f"batch_summary_search_{batch_id}",
        placeholder="item_id or job_name",
    ).strip().lower()
    status_options = sorted({str(r.get("status", "")) for r in rows if r.get("status")})
    status_filter = controls[1].multiselect(
        "Status filter",
        options=status_options,
        default=status_options,
        key=f"batch_summary_status_{batch_id}",
    )
    preferred_cols = [
        "item_id",
        "job_name",
        "status",
        "updated_at",
        # Local structural
        "complex_plddt",
        "complex_pde",
        # Global
        "ptm",
        "confidence_score",
        # Interface
        "iptm",
        "ligand_iptm",
        "protein_iptm",
        "complex_iplddt",
        "complex_ipde",
        # Affinity (shown only when present)
        "affinity",
        "ic50_uM",
        "binding_probability",
    ]
    # Keep only curated columns to avoid clutter; include key if present in any row.
    ordered_cols = [c for c in preferred_cols if any(c in r and r.get(c) is not None for r in rows)]
    # Always keep identity columns even if values are empty.
    for id_col in ("item_id", "job_name", "status", "updated_at"):
        if id_col not in ordered_cols and any(id_col in r for r in rows):
            ordered_cols.insert(min(len(ordered_cols), ("item_id", "job_name", "status", "updated_at").index(id_col)), id_col)
    sort_options = [c for c in ordered_cols if c not in {"item_id", "job_name"}]
    effective_sort_options = sort_options or ["updated_at"]
    if batch_role == "ligand" and "affinity" in effective_sort_options:
        default_sort_field = "affinity"
        default_sort_desc = False  # lower affinity value means stronger binder
    elif batch_role == "sequence" and "iptm" in effective_sort_options:
        default_sort_field = "iptm"
        default_sort_desc = True
    else:
        default_sort_field = "updated_at" if "updated_at" in effective_sort_options else effective_sort_options[0]
        default_sort_desc = True
    default_sort_index = effective_sort_options.index(default_sort_field)
    sort_field = controls[2].selectbox(
        "Sort by",
        options=effective_sort_options,
        key=f"batch_summary_sort_field_{batch_id}",
        index=default_sort_index,
    )
    sort_desc = controls[3].checkbox(
        "Desc",
        value=default_sort_desc,
        key=f"batch_summary_sort_desc_{batch_id}",
    )

    filtered_rows = []
    for row in rows:
        if status_filter and str(row.get("status", "")) not in status_filter:
            continue
        if search_query:
            hay = f"{row.get('item_id', '')} {row.get('job_name', '')}".lower()
            if search_query not in hay:
                continue
        filtered_rows.append(row)

    def _sort_key(row: dict):
        value = row.get(sort_field)
        if isinstance(value, (int, float)):
            return (0, float(value))
        if value is None:
            return (1, "")
        return (1, str(value))

    filtered_rows.sort(key=_sort_key, reverse=sort_desc)
    ordered_rows = [{k: row.get(k) for k in ordered_cols} for row in filtered_rows]

    st.dataframe(
        ordered_rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "item_id": st.column_config.TextColumn("item_id"),
            "job_name": st.column_config.TextColumn("job_name"),
            "status": st.column_config.TextColumn("status"),
            "updated_at": st.column_config.TextColumn("updated_at"),
            "complex_plddt": st.column_config.NumberColumn("complex_plddt", format="%.3f", help=METRIC_HELP.get("complex_plddt")),
            "complex_pde": st.column_config.NumberColumn("complex_pde", format="%.3f", help=METRIC_HELP.get("complex_pde")),
            "ptm": st.column_config.NumberColumn("ptm", format="%.3f", help=METRIC_HELP.get("ptm")),
            "confidence_score": st.column_config.NumberColumn("confidence_score", format="%.3f", help=METRIC_HELP.get("confidence_score")),
            "iptm": st.column_config.NumberColumn("iptm", format="%.3f", help=METRIC_HELP.get("iptm")),
            "ligand_iptm": st.column_config.NumberColumn("ligand_iptm", format="%.3f", help=METRIC_HELP.get("ligand_iptm")),
            "protein_iptm": st.column_config.NumberColumn("protein_iptm", format="%.3f", help=METRIC_HELP.get("protein_iptm")),
            "complex_iplddt": st.column_config.NumberColumn("complex_iplddt", format="%.3f", help=METRIC_HELP.get("complex_iplddt")),
            "complex_ipde": st.column_config.NumberColumn("complex_ipde", format="%.3f", help=METRIC_HELP.get("complex_ipde")),
            "affinity": st.column_config.NumberColumn("affinity", format="%.3f", help=METRIC_HELP.get("affinity")),
            "ic50_uM": st.column_config.NumberColumn("ic50_uM", format="%.4g", help=METRIC_HELP.get("ic50_uM")),
            "binding_probability": st.column_config.NumberColumn("binding_probability", format="%.3f", help=METRIC_HELP.get("binding_probability")),
        },
    )
    return filtered_rows


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
    metrics = merged_metrics_for_job(job_record)
    if metrics:
        # Curated metrics order: local structural -> global -> interface -> affinity.
        ordered_keys = [
            "complex_plddt",   # local structural confidence
            "complex_pde",     # local structural error (A)
            "ptm",             # global topology confidence
            "confidence_score",
            "iptm",            # interface confidence
            "ligand_iptm",
            "protein_iptm",
            "complex_iplddt",
            "complex_ipde",
            "affinity",        # affinity output
            "binding_probability",
        ]
        display_metrics = {}
        for key in ordered_keys:
            if key in metrics and isinstance(metrics[key], (int, float)):
                display_metrics[key] = metrics[key]
        affinity_value = display_metrics.get("affinity")
        if isinstance(affinity_value, (int, float)):
            display_metrics["ic50_uM"] = float(10**affinity_value)

        if display_metrics:
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
                metric_columns[index % len(metric_columns)].metric(
                    f"{key}",
                    display,
                    help=METRIC_HELP.get(key, ""),
                )

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
        archive_bytes = build_results_zip_bytes(job_dir)
        st.download_button(
            "Download full results",
            data=archive_bytes,
            file_name=f"{Path(job_dir).name}.zip",
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

if st.session_state["nav_page"] == "New Job":
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

    st.subheader("Batch Mode")
    batch_mode = st.selectbox(
        "Batch input type",
        ["none", "batch_ligands", "batch_sequences"],
        format_func=lambda x: {
            "none": "None (single run)",
            "batch_ligands": "Batch ligands (ID,SMILES)",
            "batch_sequences": "Batch sequences (ID,SEQUENCE)",
        }[x],
        key="batch_mode",
    )
    batch_file = None
    batch_items: list[tuple[str, str]] = []
    if st.session_state.get("last_batch_mode") != batch_mode:
        on_batch_mode_change()
        st.session_state["last_batch_mode"] = batch_mode
        st.rerun()
    else:
        st.session_state["last_batch_mode"] = batch_mode
    if batch_mode != "none":
        batch_text = st.text_area(
            "Batch text input (ID,VALUE)",
            key="batch_text_input",
            height=120,
            placeholder="LIG_001,CC(=O)OC1=CC=CC=C1C(=O)O",
        )
        batch_file = st.file_uploader(
            "Upload batch file (.txt/.csv)",
            type=["txt", "csv", "tsv"],
            key="batch_file",
            help="Each line: ID,VALUE",
        )
        if batch_text.strip():
            batch_items = parse_batch_items(batch_text)
        elif batch_file is not None:
            try:
                text = batch_file.getvalue().decode("utf-8", errors="ignore")
            except Exception:
                text = ""
            batch_items = parse_batch_items(text)
            if batch_items:
                st.caption(f"Parsed {len(batch_items)} batch item(s).")
            else:
                st.warning("No valid batch rows parsed. Expected lines like: ID,VALUE")

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
        st.button("Load example", on_click=load_context_example, args=(batch_mode,), use_container_width=True)

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
            preview_entities = deepcopy(normalized_entities)
            preview_sequence = first_protein_sequence or ""
            if batch_mode == "batch_ligands" and batch_items:
                first_batch_smiles = batch_items[0][1]
                ligand_idx = next((i for i, e in enumerate(preview_entities) if e["type"] == "ligand"), None)
                if ligand_idx is not None:
                    preview_entities[ligand_idx]["input"] = first_batch_smiles
                else:
                    preview_entities.append(
                        {
                            "type": "ligand",
                            "copies": 1,
                            "input": first_batch_smiles,
                            "use_affinity": True,
                            "cyclic": False,
                        }
                    )
            elif batch_mode == "batch_sequences" and batch_items:
                first_batch_seq = batch_items[0][1]
                preview_entities.append(
                    {
                        "type": "protein",
                        "copies": 1,
                        "input": first_batch_seq,
                        "use_affinity": False,
                        "cyclic": False,
                    }
                )
                preview_sequence = first_protein_sequence or ""
            yaml_preview = build_boltz_yaml_text(
                sequence=preview_sequence,
                entities=preview_entities,
                msa_path=yaml_msa_path,
                enable_affinity=runtime_settings["enable_affinity"],
            )
            st.code(yaml_preview, language="yaml")
            if batch_mode != "none" and batch_items:
                st.caption(f"Batch will generate {len(batch_items)} YAML/job inputs.")
                with st.expander("Show first batch YAMLs", expanded=False):
                    max_preview = min(5, len(batch_items))
                    for idx in range(max_preview):
                        item_id, item_value = batch_items[idx]
                        entities_item = deepcopy(normalized_entities)
                        seq_item = first_protein_sequence or ""
                        if batch_mode == "batch_ligands":
                            lig_idx = next((i for i, e in enumerate(entities_item) if e["type"] == "ligand"), None)
                            if lig_idx is None:
                                entities_item.append(
                                    {
                                        "type": "ligand",
                                        "copies": 1,
                                        "input": item_value,
                                        "use_affinity": True,
                                        "cyclic": False,
                                    }
                                )
                            else:
                                entities_item[lig_idx]["input"] = item_value
                                entities_item[lig_idx]["use_affinity"] = True
                        else:
                            entities_item.append(
                                {
                                    "type": "protein",
                                    "copies": 1,
                                    "input": item_value,
                                    "use_affinity": False,
                                    "cyclic": False,
                                }
                            )
                            seq_item = first_protein_sequence or ""
                        yaml_item = build_boltz_yaml_text(
                            sequence=seq_item,
                            entities=entities_item,
                            msa_path=yaml_msa_path,
                            enable_affinity=runtime_settings["enable_affinity"],
                        )
                        st.markdown(f"**{idx + 1}. {item_id}**")
                        st.code(yaml_item, language="yaml")

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
        elif batch_mode != "none" and not batch_items:
            st.error("Batch mode selected, but no valid batch items were found.")
        else:
            base_name = st.session_state["job_name"].strip()
            if batch_mode == "none":
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
                    base_name,
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
            else:
                batch_id = str(uuid.uuid4())
                batch_role = "ligand" if batch_mode == "batch_ligands" else "sequence"
                queued_ids: list[str] = []
                for item_id, item_value in batch_items:
                    entities_item = deepcopy(normalized_entities)
                    if batch_role == "ligand":
                        ligand_idx = next((i for i, e in enumerate(entities_item) if e["type"] == "ligand"), None)
                        if ligand_idx is None:
                            entities_item.append(
                                {
                                    "type": "ligand",
                                    "copies": 1,
                                    "input": item_value,
                                    "use_affinity": True,
                                    "cyclic": False,
                                }
                            )
                        else:
                            entities_item[ligand_idx]["input"] = item_value
                            entities_item[ligand_idx]["use_affinity"] = True
                    else:
                        entities_item.append(
                            {
                                "type": "protein",
                                "copies": 1,
                                "input": item_value,
                                "use_affinity": False,
                                "cyclic": False,
                            }
                        )
                    sequence_for_hash = next((e["input"] for e in entities_item if e["type"] == "protein"), "")
                    candidate_yaml = build_boltz_yaml_text(
                        sequence=sequence_for_hash,
                        entities=entities_item,
                        msa_path=None,
                        enable_affinity=runtime_settings["enable_affinity"],
                    )
                    candidate_hash = compute_yaml_hash(candidate_yaml)
                    store_input_yaml_snapshot(runtime_settings["cache_dir"], candidate_hash, candidate_yaml)
                    job_id = queue_job(
                        f"{base_name}_{item_id}",
                        entities_item,
                        runtime_settings,
                        input_hash=candidate_hash,
                        batch_id=batch_id,
                        batch_name=base_name,
                        batch_item_id=item_id,
                        batch_role=batch_role,
                    )
                    queued_ids.append(job_id)
                spawned_worker = start_worker_if_needed()
                if queued_ids:
                    st.success(f"Queued {len(queued_ids)} batch job(s). Batch ID: {batch_id}")
                    if spawned_worker:
                        st.caption("Background worker started.")
                else:
                    st.error("No batch jobs were queued. Ensure the matching entity type exists in Entities.")

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
    grouped_batches = batch_groups(jobs)
    if grouped_batches:
        st.subheader("Batch Runs")
        BATCHES_PER_PAGE = 5
        total_batches = len(grouped_batches)
        total_batch_pages = max(1, (total_batches + BATCHES_PER_PAGE - 1) // BATCHES_PER_PAGE)
        current_batch_page = int(st.session_state.get("batch_page", 1))
        current_batch_page = max(1, min(current_batch_page, total_batch_pages))
        st.session_state["batch_page"] = current_batch_page
        start_batch_idx = (current_batch_page - 1) * BATCHES_PER_PAGE
        end_batch_idx = start_batch_idx + BATCHES_PER_PAGE
        visible_batches = grouped_batches[start_batch_idx:end_batch_idx]

        bp1, bp2, bp3 = st.columns([1, 2, 1])
        with bp1:
            if st.button("Previous batches", use_container_width=True, disabled=current_batch_page <= 1):
                st.session_state["batch_page"] = current_batch_page - 1
                st.rerun()
        with bp2:
            st.caption(f"Batch page {current_batch_page}/{total_batch_pages} ({total_batches} total)")
        with bp3:
            if st.button("Next batches", use_container_width=True, disabled=current_batch_page >= total_batch_pages):
                st.session_state["batch_page"] = current_batch_page + 1
                st.rerun()

        batch_ids_visible = [b["batch_id"] for b in visible_batches]
        batch_actions_c1, batch_actions_c2 = st.columns(2)
        with batch_actions_c1:
            st.button(
                "Select all batches",
                use_container_width=True,
                on_click=select_all_filtered_batches,
                args=(batch_ids_visible,),
                disabled=not batch_ids_visible,
            )
        with batch_actions_c2:
            st.button(
                "Deselect all batches",
                use_container_width=True,
                on_click=deselect_all_filtered_batches,
                args=(batch_ids_visible,),
                disabled=not batch_ids_visible,
            )

        selected_batches_now = set(st.session_state.get("selected_batch_ids", []))
        for batch in visible_batches:
            cols = st.columns([0.7, 4, 2, 2, 2, 1])
            is_selected = batch["batch_id"] in selected_batches_now
            cols[0].checkbox(
                "Select batch",
                value=is_selected,
                key=f"batch_select_{batch['batch_id']}",
                label_visibility="collapsed",
            )
            if st.session_state.get(f"batch_select_{batch['batch_id']}", False):
                selected_batches_now.add(batch["batch_id"])
            else:
                selected_batches_now.discard(batch["batch_id"])
            cols[1].write(f"**{batch['batch_name']}**")
            cols[2].write(batch["status"])
            cols[3].write(f"{batch['total']} jobs")
            cols[4].write(batch.get("updated_at", "-"))
            if cols[5].button("Open", key=f"open_batch_{batch['batch_id']}"):
                st.session_state["selected_batch_id"] = batch["batch_id"]
                st.session_state["nav_page"] = "Batch Details"
                st.rerun()
            st.caption(
                f"queued={batch['counts'].get('queued',0)} | running={batch['counts'].get('running',0)} | "
                f"completed={batch['counts'].get('completed',0)} | failed={batch['counts'].get('failed',0)}"
            )
        st.session_state["selected_batch_ids"] = list(selected_batches_now)

        st.button(
            "Delete selected batches",
            type="secondary",
            use_container_width=True,
            disabled=not st.session_state.get("selected_batch_ids", []),
            on_click=lambda: st.session_state.update(
                {"pending_delete_batch_ids": list(st.session_state.get("selected_batch_ids", []))}
            ),
        )

        pending_batch_ids = list(st.session_state.get("pending_delete_batch_ids", []))
        if pending_batch_ids:
            pending_batches = [b for b in grouped_batches if b["batch_id"] in set(pending_batch_ids)]
            st.warning("Delete selected batches?")
            for pb in pending_batches:
                st.write(f"- {pb['batch_name']} ({pb['total']} jobs, {pb['status']})")
            d1, d2 = st.columns(2)
            with d1:
                if st.button("Confirm selected batch delete", type="primary", use_container_width=True):
                    removed_count, errors = remove_selected_batches(
                        pending_batch_ids,
                        runtime_settings["cache_dir"],
                        runtime_settings["results_dir"],
                    )
                    st.session_state["selected_batch_ids"] = [
                        bid for bid in st.session_state.get("selected_batch_ids", []) if bid not in set(pending_batch_ids)
                    ]
                    st.session_state["pending_delete_batch_ids"] = []
                    if removed_count:
                        st.session_state["jobs_flash"] = f"Removed {removed_count} jobs from selected batches."
                    if errors:
                        st.session_state["jobs_flash"] = (
                            st.session_state.get("jobs_flash", "")
                            + ("; " if st.session_state.get("jobs_flash") else "")
                            + f"Batch delete errors: {'; '.join(errors)}"
                        )
                    st.rerun()
            with d2:
                if st.button("Cancel selected batch delete", use_container_width=True):
                    st.session_state["pending_delete_batch_ids"] = []
                    st.rerun()

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
    filtered_inline = sorted(filtered_inline, key=lambda j: j.get("name", ""), reverse=True)
    JOBS_PER_PAGE = 15
    total_filtered_jobs = len(filtered_inline)
    total_job_pages = max(1, (total_filtered_jobs + JOBS_PER_PAGE - 1) // JOBS_PER_PAGE)
    current_jobs_page = int(st.session_state.get("jobs_page", 1))
    current_jobs_page = max(1, min(current_jobs_page, total_job_pages))
    st.session_state["jobs_page"] = current_jobs_page
    start_job_idx = (current_jobs_page - 1) * JOBS_PER_PAGE
    end_job_idx = start_job_idx + JOBS_PER_PAGE
    visible_jobs = filtered_inline[start_job_idx:end_job_idx]
    filtered_ids = [j["id"] for j in visible_jobs]

    jp1, jp2, jp3 = st.columns([1, 2, 1])
    with jp1:
        if st.button("Previous jobs", use_container_width=True, disabled=current_jobs_page <= 1):
            st.session_state["jobs_page"] = current_jobs_page - 1
            st.rerun()
    with jp2:
        st.caption(f"Job page {current_jobs_page}/{total_job_pages} ({total_filtered_jobs} filtered)")
    with jp3:
        if st.button("Next jobs", use_container_width=True, disabled=current_jobs_page >= total_job_pages):
            st.session_state["jobs_page"] = current_jobs_page + 1
            st.rerun()

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


    if not visible_jobs:
        st.caption("No jobs found.")
    selected_now = set(st.session_state.get("selected_job_ids", []))
    for job in visible_jobs:
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
            st.session_state["job_return_page"] = "New Job"
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

elif st.session_state["nav_page"] == "Job Details":
    jobs = all_jobs_with_legacy(runtime_settings["results_dir"])
    selected_id = st.session_state.get("selected_job_id")
    job = next((j for j in jobs if j["id"] == selected_id), None)
    top_cols = st.columns([1, 4])
    if st.session_state.get("job_return_page") == "Batch Details" and st.session_state.get("selected_batch_id"):
        if top_cols[0].button("Back to Batch"):
            st.session_state["nav_page"] = "Batch Details"
            st.rerun()
    else:
        if top_cols[0].button("Back to New Job"):
            st.session_state["nav_page"] = "New Job"
            st.rerun()
    if job is None:
        st.warning("No job selected.")
    else:
        render_result_view(job)
else:
    jobs = all_jobs_with_legacy(runtime_settings["results_dir"])
    batch_id = st.session_state.get("selected_batch_id")
    members = [j for j in jobs if j.get("batch_id") == batch_id]
    top_cols = st.columns([1, 4])
    if top_cols[0].button("Back to New Job"):
        st.session_state["nav_page"] = "New Job"
        st.rerun()
    if not members:
        st.warning("No batch selected.")
    else:
        members = sorted(members, key=lambda j: j.get("name", ""))
        st.subheader(f"Batch: {members[0].get('batch_name', batch_id)}")
        st.caption(f"Batch ID: {batch_id} • {len(members)} job(s)")
        filtered_rows = render_ligand_batch_summary(batch_id, members)
        member_by_id = {j.get("batch_item_id") or j.get("name", ""): j for j in members}
        list_source = [member_by_id.get(r.get("item_id")) for r in filtered_rows] if filtered_rows else members
        list_source = [j for j in list_source if j is not None]
        st.markdown("**Batch Items**")
        for job in list_source:
            cols = st.columns([3, 2, 2, 1])
            label = job.get("batch_item_id") or job["name"]
            cols[0].write(f"**{label}**")
            cols[1].write(job.get("status", "-"))
            cols[2].write(job.get("updated_at", "-"))
            if cols[3].button("Open", key=f"batch_member_open_{job['id']}"):
                st.session_state["selected_job_id"] = job["id"]
                st.session_state["job_return_page"] = "Batch Details"
                st.session_state["nav_page"] = "Job Details"
                st.rerun()
