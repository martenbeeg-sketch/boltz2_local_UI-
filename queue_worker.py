import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from boltz_runner import run_prediction


APP_DIR = Path(__file__).resolve().parent
JOBS_DB_PATH = APP_DIR / "jobs_db.json"
WORKER_PID_PATH = APP_DIR / "queue_worker.pid"
WORKER_LOCK_PATH = APP_DIR / "queue_worker.lock"
EVENT_SERVER_PORT = 8766
EVENT_SERVER_PUBLISH_URL = f"http://127.0.0.1:{EVENT_SERVER_PORT}/publish"


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def load_jobs_db() -> list[dict]:
    if not JOBS_DB_PATH.exists():
        return []
    try:
        data = json.loads(JOBS_DB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def save_jobs_db(jobs: list[dict]) -> None:
    JOBS_DB_PATH.write_text(json.dumps(jobs, indent=2), encoding="utf-8")


def notify_status_event(event: str, job_id: str, status: str, name: str) -> None:
    payload = json.dumps(
        {"event": event, "job_id": job_id, "status": status, "name": name, "at": utc_now()},
        separators=(",", ":"),
    )
    try:
        url = f"{EVENT_SERVER_PUBLISH_URL}?{urlencode({'event': payload})}"
        urlopen(url, timeout=1.0).read()
    except Exception:
        pass


def claim_next_job() -> dict | None:
    jobs = load_jobs_db()
    queued_index = next((idx for idx, j in enumerate(jobs) if j.get("status") == "queued"), None)
    if queued_index is None:
        return None
    jobs[queued_index]["status"] = "running"
    jobs[queued_index]["updated_at"] = utc_now()
    save_jobs_db(jobs)
    claimed = jobs[queued_index]
    notify_status_event("running", claimed["id"], "running", claimed.get("name", claimed["id"]))
    return claimed


def finalize_job(job_id: str, result) -> None:
    jobs = load_jobs_db()
    idx = next((i for i, j in enumerate(jobs) if j.get("id") == job_id), None)
    if idx is None:
        return
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


def run_job(job: dict):
    settings = job["settings"]
    run_name = (job.get("run_name") or job.get("name") or "job").strip()
    return run_prediction(
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


def main() -> None:
    if not WORKER_LOCK_PATH.exists():
        try:
            fd = os.open(str(WORKER_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except Exception:
            return
    WORKER_PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    try:
        while True:
            job = claim_next_job()
            if job is None:
                break
            result = run_job(job)
            finalize_job(job["id"], result)
            time.sleep(0.2)
    finally:
        if WORKER_PID_PATH.exists():
            try:
                WORKER_PID_PATH.unlink()
            except Exception:
                pass
        if WORKER_LOCK_PATH.exists():
            try:
                WORKER_LOCK_PATH.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
