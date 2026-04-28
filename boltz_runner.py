import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
VALID_DNA_BASES = set("ACGTN")
VALID_RNA_BASES = set("ACGUN")
MIN_SEQ_LENGTH = 10
MAX_SEQ_LENGTH = 2500
DEFAULT_BOLTZ_CACHE_DIR = "/mnt/db/reference_files/boltz_models"
DEFAULT_DOCKER_ARGS = "--ipc=host --shm-size=48G"
DEFAULT_RESULTS_DIR = str(Path(__file__).resolve().parent / "results")
DEFAULT_MSA_REPOSITORY_DIR = str(Path(DEFAULT_BOLTZ_CACHE_DIR) / "msa_repository")
A3M_SEQUENCE_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-")


@dataclass
class PredictionResult:
    success: bool
    message: str
    job_dir: str
    structure_path: Optional[str] = None
    structure_text: Optional[str] = None
    structure_format: Optional[str] = None
    metrics: Optional[dict] = None
    raw_log: str = ""
    plddt: Optional[np.ndarray] = None
    pae: Optional[np.ndarray] = None


def parse_fasta(text: str) -> tuple[str, str]:
    lines = text.strip().splitlines()
    header = ""
    seq_parts: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:].strip()
        else:
            seq_parts.append("".join(ch for ch in line if ch.isalpha()).upper())
    sequence = "".join(seq_parts)
    if not header and not any(line.startswith(">") for line in lines):
        header = "protein"
        sequence = "".join(ch for ch in text if ch.isalpha()).upper()
    return header[:30] or "protein", sequence


def validate_protein(sequence: str) -> tuple[bool, str]:
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    invalid = set(seq) - VALID_AMINO_ACIDS
    if invalid:
        return False, f"Invalid amino acids: {', '.join(sorted(invalid))}"
    if len(seq) < MIN_SEQ_LENGTH:
        return False, f"Sequence too short. Minimum is {MIN_SEQ_LENGTH} residues."
    if len(seq) > MAX_SEQ_LENGTH:
        return False, f"Sequence too long ({len(seq)} aa). Maximum is {MAX_SEQ_LENGTH}."
    return True, seq


def validate_smiles(smiles: str) -> tuple[bool, str]:
    if not smiles or not smiles.strip():
        return True, ""
    smiles = smiles.strip()
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz[]()=#@+-.0123456789\\/,:")
    invalid = set(smiles) - allowed
    if invalid:
        return False, f"Invalid SMILES characters: {', '.join(sorted(invalid))}"
    return True, smiles


def validate_dna(sequence: str) -> tuple[bool, str]:
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    invalid = set(seq) - VALID_DNA_BASES
    if invalid:
        return False, f"Invalid DNA bases: {', '.join(sorted(invalid))}"
    if len(seq) < 2:
        return False, "DNA sequence too short."
    return True, seq


def validate_rna(sequence: str) -> tuple[bool, str]:
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    invalid = set(seq) - VALID_RNA_BASES
    if invalid:
        return False, f"Invalid RNA bases: {', '.join(sorted(invalid))}"
    if len(seq) < 2:
        return False, "RNA sequence too short."
    return True, seq


def parse_entities(entities: list[dict]) -> tuple[list[dict], Optional[str], str]:
    normalized: list[dict] = []
    first_protein_sequence: Optional[str] = None
    first_protein_header = "protein"

    for item in entities:
        entity_type = str(item.get("type", "protein")).strip().lower()
        copies = int(item.get("copies", 1) or 1)
        raw_input = str(item.get("input", "") or "").strip()
        use_affinity = bool(item.get("use_affinity", False))
        cyclic = bool(item.get("cyclic", False))
        if copies < 1:
            return [], None, "Copies must be >= 1 for all entities."
        if not raw_input:
            return [], None, f"Input is required for entity type '{entity_type}'."

        if entity_type == "protein":
            header, seq = parse_fasta(raw_input)
            ok, seq_or_error = validate_protein(seq)
            if not ok:
                return [], None, seq_or_error
            clean_input = seq_or_error
            if first_protein_sequence is None:
                first_protein_sequence = clean_input
                first_protein_header = header
        elif entity_type == "dna":
            ok, seq_or_error = validate_dna(raw_input)
            if not ok:
                return [], None, seq_or_error
            clean_input = seq_or_error
        elif entity_type == "rna":
            ok, seq_or_error = validate_rna(raw_input)
            if not ok:
                return [], None, seq_or_error
            clean_input = seq_or_error
        elif entity_type in {"ligand", "ion"}:
            ok, smiles_or_error = validate_smiles(raw_input)
            if not ok:
                return [], None, smiles_or_error
            clean_input = smiles_or_error
        else:
            return [], None, f"Unsupported entity type: {entity_type}"

        normalized.append(
            {
                "type": entity_type,
                "copies": copies,
                "input": clean_input,
                "use_affinity": use_affinity,
                "cyclic": cyclic,
            }
        )

    if first_protein_sequence is None:
        return [], None, "At least one protein entity is required."
    return normalized, first_protein_sequence, first_protein_header


def create_boltz_yaml(
    sequence: str,
    output_dir: str,
    *,
    ligand_smiles: Optional[str] = None,
    entities: Optional[list[dict]] = None,
    msa_path: Optional[str] = None,
    enable_affinity: bool = True,
    num_copies: int = 1,
    cyclic: bool = False,
) -> str:
    yaml_text = build_boltz_yaml_text(
        sequence=sequence,
        ligand_smiles=ligand_smiles,
        entities=entities,
        msa_path=msa_path,
        enable_affinity=enable_affinity,
        num_copies=num_copies,
        cyclic=cyclic,
    )
    path = os.path.join(output_dir, "input.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(yaml_text)
    return path


def build_boltz_yaml_text(
    *,
    sequence: str,
    ligand_smiles: Optional[str] = None,
    entities: Optional[list[dict]] = None,
    msa_path: Optional[str] = None,
    enable_affinity: bool = True,
    num_copies: int = 1,
    cyclic: bool = False,
) -> str:
    if entities:
        return build_boltz_yaml_text_from_entities(
            entities=entities,
            msa_path=msa_path,
            global_enable_affinity=enable_affinity,
        )

    chain_ids = [chr(ord("A") + idx) for idx in range(num_copies)]
    lines = ["version: 1", "sequences:", "  - protein:"]
    if num_copies > 1:
        id_list = ", ".join(f'"{chain_id}"' for chain_id in chain_ids)
        lines.append(f"      id: [{id_list}]")
    else:
        lines.append('      id: "A"')
    lines.append(f"      sequence: {sequence}")
    if msa_path:
        lines.append(f"      msa: {msa_path}")
    if cyclic:
        lines.append("      cyclic: true")
    if ligand_smiles:
        ligand_id = chr(ord("A") + num_copies)
        lines.extend(
            [
                "  - ligand:",
                f'      id: "{ligand_id}"',
                f'      smiles: "{ligand_smiles}"',
            ]
        )
        if enable_affinity:
            lines.extend(
                [
                    "properties:",
                    "  - affinity:",
                    f'      binder: "{ligand_id}"',
                ]
            )
    return "\n".join(lines) + "\n"


def build_boltz_yaml_text_from_entities(
    *,
    entities: list[dict],
    msa_path: Optional[str],
    global_enable_affinity: bool,
) -> str:
    lines = ["version: 1", "sequences:"]
    next_chain_idx = 0
    affinity_ids: list[str] = []
    msa_assigned = False

    for entity in entities:
        entity_type = entity["type"]
        copies = int(entity["copies"])
        input_value = entity["input"]
        use_affinity = bool(entity.get("use_affinity", False))
        cyclic = bool(entity.get("cyclic", False))

        chain_ids = [chr(ord("A") + next_chain_idx + idx) for idx in range(copies)]
        next_chain_idx += copies
        if copies > 1:
            id_field = "[" + ", ".join(f'"{cid}"' for cid in chain_ids) + "]"
        else:
            id_field = f'"{chain_ids[0]}"'

        yaml_type = "ligand" if entity_type == "ion" else entity_type
        lines.append(f"  - {yaml_type}:")
        lines.append(f"      id: {id_field}")

        if yaml_type in {"protein", "dna", "rna"}:
            lines.append(f"      sequence: {input_value}")
            if yaml_type == "protein" and msa_path and not msa_assigned:
                lines.append(f"      msa: {msa_path}")
                msa_assigned = True
            if yaml_type == "protein" and cyclic:
                lines.append("      cyclic: true")
        else:
            lines.append(f'      smiles: "{input_value}"')
            if global_enable_affinity and use_affinity:
                affinity_ids.extend(chain_ids)

    if affinity_ids:
        lines.append("properties:")
        for ligand_id in affinity_ids:
            lines.extend(
                [
                    "  - affinity:",
                    f'      binder: "{ligand_id}"',
                ]
            )

    return "\n".join(lines) + "\n"


def run_prediction(
    protein_text: str,
    ligand_smiles: str,
    entities: Optional[list[dict]] = None,
    *,
    job_name: str,
    results_dir: str,
    cache_dir: str,
    msa_repository_dir: str,
    docker_image: str,
    docker_args: str,
    use_msa_repository: bool,
    use_potentials: bool,
    enable_affinity: bool,
    sampling_steps: int,
    recycling_steps: int,
    diffusion_samples: int,
    sampling_steps_affinity: int,
    diffusion_samples_affinity: int,
    affinity_mw_correction: bool,
    num_copies: int,
    cyclic: bool,
    use_msa_server: bool,
    gpu_device: str,
) -> PredictionResult:
    if not job_name or not job_name.strip():
        return PredictionResult(False, "Job name is required.", job_dir="")

    if entities:
        normalized_entities, first_protein_sequence, header_or_error = parse_entities(entities)
        if not normalized_entities:
            return PredictionResult(False, header_or_error, job_dir="")
        sequence = first_protein_sequence or ""
        header = header_or_error
        ligand_smiles_present = any(e["type"] in {"ligand", "ion"} for e in normalized_entities)
        entities_for_yaml = normalized_entities
    else:
        header, sequence = parse_fasta(protein_text)
        ok, protein_or_error = validate_protein(sequence)
        if not ok:
            return PredictionResult(False, protein_or_error, job_dir="")

        ok, smiles_or_error = validate_smiles(ligand_smiles)
        if not ok:
            return PredictionResult(False, smiles_or_error, job_dir="")

        sequence = protein_or_error
        ligand_smiles = smiles_or_error or None
        ligand_smiles_present = bool(ligand_smiles)
        entities_for_yaml = None
    cache_dir = cache_dir or os.getenv("BOLTZ_CACHE_DIR", DEFAULT_BOLTZ_CACHE_DIR)
    msa_repository_dir = (
        msa_repository_dir
        or os.getenv("BOLTZ_MSA_REPOSITORY_DIR", DEFAULT_MSA_REPOSITORY_DIR)
    )

    cached_msa_host, cached_msa_container = get_cached_msa_paths(
        sequence=sequence, msa_repository_dir=msa_repository_dir
    )
    use_cached_msa = False
    msa_cache_note = ""
    if use_msa_repository and os.path.exists(cached_msa_host):
        valid_msa, msa_reason = is_valid_a3m_file(cached_msa_host)
        if not valid_msa:
            repaired, repair_reason = try_repair_a3m_file(cached_msa_host)
            if repaired:
                valid_msa = True
                msa_reason = ""
                msa_cache_note = f"MSA cache entry repaired and reused: {cached_msa_host}"
            else:
                msa_reason = repair_reason or msa_reason
        if valid_msa:
            use_cached_msa = True
        else:
            quarantine_path = quarantine_invalid_msa(cached_msa_host)
            if quarantine_path:
                msa_cache_note = (
                    "MSA cache entry was invalid and quarantined: "
                    f"{quarantine_path} ({msa_reason})"
                )
            else:
                msa_cache_note = (
                    "MSA cache entry was invalid but could not be quarantined: "
                    f"{cached_msa_host} ({msa_reason})"
                )
    effective_use_msa_server = bool(use_msa_server and not use_cached_msa)
    yaml_msa_path = cached_msa_container if use_cached_msa else None

    job_dir = create_job_dir(job_name or header, results_dir)
    yaml_path = create_boltz_yaml(
        sequence,
        job_dir,
        ligand_smiles=ligand_smiles,
        entities=entities_for_yaml,
        msa_path=yaml_msa_path,
        enable_affinity=enable_affinity,
        num_copies=int(num_copies),
        cyclic=cyclic,
    )

    command = build_docker_command(
        yaml_path=yaml_path,
        output_dir=job_dir,
        cache_dir=cache_dir,
        msa_repository_dir=msa_repository_dir,
        docker_image=docker_image,
        docker_args=docker_args,
        use_potentials=use_potentials,
        sampling_steps=sampling_steps,
        recycling_steps=recycling_steps,
        diffusion_samples=diffusion_samples,
        sampling_steps_affinity=sampling_steps_affinity,
        diffusion_samples_affinity=diffusion_samples_affinity,
        affinity_mw_correction=affinity_mw_correction,
        use_msa_server=effective_use_msa_server,
        gpu_device=gpu_device,
    )

    raw_log = "Command:\n" + shlex.join(command) + "\n\n"
    if use_cached_msa:
        raw_log += (
            f"MSA cache hit: using {cached_msa_host}\n"
            "Skipping MSA server for this run.\n\n"
        )
    elif use_msa_server and use_msa_repository:
        raw_log += (
            f"MSA cache miss: will query server and then store under {cached_msa_host}\n\n"
        )
    if msa_cache_note:
        raw_log += f"{msa_cache_note}\n\n"
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("BOLTZ_TIMEOUT_SECONDS", "1800")),
        )
        raw_log += f"Return code: {completed.returncode}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}\n"
    except subprocess.TimeoutExpired:
        return PredictionResult(False, "Prediction timed out.", job_dir=job_dir, raw_log=raw_log)
    except FileNotFoundError:
        return PredictionResult(False, "Docker is not installed or not on PATH.", job_dir=job_dir, raw_log=raw_log)
    except Exception as exc:
        return PredictionResult(False, f"Failed to run Docker: {exc}", job_dir=job_dir, raw_log=raw_log)

    structure_path = find_structure_path(job_dir)
    if not structure_path:
        message = extract_error(completed.stderr or completed.stdout or "No structure output found.")
        return PredictionResult(False, message, job_dir=job_dir, raw_log=raw_log)

    if use_msa_server and use_msa_repository and not use_cached_msa:
        save_generated_msa_to_repository(job_dir, cached_msa_host)
        valid_msa, msa_reason = is_valid_a3m_file(cached_msa_host)
        if os.path.exists(cached_msa_host) and valid_msa:
            raw_log += f"\nMSA cached to: {cached_msa_host}\n"
        elif os.path.exists(cached_msa_host):
            quarantine_path = quarantine_invalid_msa(cached_msa_host)
            if quarantine_path:
                raw_log += (
                    "\nMSA caching note: generated .a3m was invalid, quarantined to: "
                    f"{quarantine_path} ({msa_reason})\n"
                )
            else:
                raw_log += (
                    "\nMSA caching note: generated .a3m was invalid and could not be "
                    f"quarantined ({msa_reason}).\n"
                )
        else:
            raw_log += "\nMSA caching note: no generated .a3m file found to store.\n"

    structure_text = Path(structure_path).read_text(encoding="utf-8")
    confidence_files = find_confidence_files(job_dir)
    metrics = collect_metrics(confidence_files["json_files"])
    expects_affinity = bool(ligand_smiles_present and enable_affinity)
    affinity_found = "affinity" in metrics or "binding_probability" in metrics
    status_message = f"Prediction complete for {len(sequence)} aa sequence ({header})."
    if expects_affinity and not affinity_found:
        status_message += " Affinity metrics were requested but not found in output JSON."
    return PredictionResult(
        True,
        status_message,
        job_dir=job_dir,
        structure_path=structure_path,
        structure_text=structure_text,
        structure_format="cif" if structure_path.endswith(".cif") else "pdb",
        metrics=metrics,
        raw_log=raw_log,
        plddt=load_plddt(confidence_files["plddt"], confidence_files["confidence_json"], structure_path),
        pae=load_pae(confidence_files["pae"], confidence_files["confidence_json"]),
    )


def create_job_dir(job_name: str, results_dir: str) -> str:
    results_root = Path(results_dir or os.getenv("BOLTZ_RESULTS_DIR", DEFAULT_RESULTS_DIR))
    results_root.mkdir(parents=True, exist_ok=True)
    safe_header = re.sub(r"[^A-Za-z0-9_-]+", "_", job_name).strip("_") or "protein"
    if re.match(r"^\d{8}_\d{6}_[A-Za-z0-9_-]+$", safe_header):
        base_name = safe_header
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{timestamp}_{safe_header[:30]}"
    base_dir = results_root / base_name
    candidate = base_dir
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = results_root / f"{base_dir.name}_{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return str(candidate)


def build_docker_command(
    *,
    yaml_path: str,
    output_dir: str,
    cache_dir: str,
    msa_repository_dir: str,
    docker_image: str,
    docker_args: str,
    use_potentials: bool,
    sampling_steps: int,
    recycling_steps: int,
    diffusion_samples: int,
    sampling_steps_affinity: int,
    diffusion_samples_affinity: int,
    affinity_mw_correction: bool,
    use_msa_server: bool,
    gpu_device: str,
) -> list[str]:
    image = docker_image or os.getenv("BOLTZ_DOCKER_IMAGE", "ovoex-boltz2")
    cache_dir = cache_dir or os.getenv("BOLTZ_CACHE_DIR", DEFAULT_BOLTZ_CACHE_DIR)
    msa_repository_dir = (
        msa_repository_dir
        or os.getenv("BOLTZ_MSA_REPOSITORY_DIR", DEFAULT_MSA_REPOSITORY_DIR)
    )
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(msa_repository_dir, exist_ok=True)
    extra_args = shlex.split(docker_args or os.getenv("BOLTZ_DOCKER_ARGS", DEFAULT_DOCKER_ARGS))
    container_workdir = "/work"
    gpu_request = "all" if gpu_device == "all" else f"device={gpu_device}"
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        gpu_request,
        "-v",
        f"{os.path.abspath(output_dir)}:{container_workdir}",
        "-v",
        f"{os.path.abspath(cache_dir)}:/cache",
        "-v",
        f"{os.path.abspath(msa_repository_dir)}:/msa_repository",
        "-e",
        "BOLTZ_CACHE=/cache",
        "-e",
        f"CUDA_VISIBLE_DEVICES={'0' if gpu_device != 'all' else 'all'}",
    ]
    command.extend(extra_args)
    command.extend(
        [
            image,
            "predict",
            f"{container_workdir}/{os.path.basename(yaml_path)}",
            "--out_dir",
            container_workdir,
            "--sampling_steps",
            str(sampling_steps),
            "--recycling_steps",
            str(recycling_steps),
            "--diffusion_samples",
            str(diffusion_samples),
            "--sampling_steps_affinity",
            str(sampling_steps_affinity),
            "--diffusion_samples_affinity",
            str(diffusion_samples_affinity),
            "--accelerator",
            "gpu",
            "--override",
        ]
    )
    if use_msa_server:
        command.append("--use_msa_server")
    if use_potentials:
        command.append("--use_potentials")
    if affinity_mw_correction:
        command.append("--affinity_mw_correction")
    return command


def find_structure_path(base_dir: str) -> Optional[str]:
    patterns = [
        os.path.join(base_dir, "boltz_results_*", "predictions", "**", "*.cif"),
        os.path.join(base_dir, "boltz_results_*", "predictions", "**", "*.pdb"),
        os.path.join(base_dir, "predictions", "**", "*.cif"),
        os.path.join(base_dir, "predictions", "**", "*.pdb"),
    ]
    for pattern in patterns:
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits[0]
    return None


def get_cached_msa_paths(sequence: str, msa_repository_dir: str) -> tuple[str, str]:
    seq_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    file_name = f"{seq_hash}.a3m"
    host_path = str(Path(msa_repository_dir) / file_name)
    container_path = f"/msa_repository/{file_name}"
    return host_path, container_path


def is_valid_a3m_file(path: str) -> tuple[bool, str]:
    file_path = Path(path)
    if not file_path.exists():
        return False, "file does not exist"
    try:
        raw = file_path.read_bytes()
    except Exception as exc:
        return False, f"read failed: {exc}"
    return validate_a3m_bytes(raw)


def validate_a3m_bytes(raw: bytes) -> tuple[bool, str]:
    if not raw:
        return False, "file is empty"
    if b"\x00" in raw:
        return False, "contains NUL byte(s)"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return False, f"invalid UTF-8: {exc}"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False, "contains no non-empty lines"
    if not lines[0].startswith(">"):
        return False, "first non-empty line is not a FASTA header"
    seen_sequence = False
    for line in lines:
        if line.startswith(">"):
            continue
        seen_sequence = True
        if any(ch not in A3M_SEQUENCE_ALLOWED for ch in line):
            return False, "contains invalid sequence characters"
    if not seen_sequence:
        return False, "contains headers only and no sequence"
    return True, ""


def sanitize_a3m_bytes(raw: bytes) -> bytes:
    # Boltz-generated A3M files can include a trailing NUL byte.
    cleaned = raw.rstrip(b"\x00")
    cleaned = cleaned.replace(b"\r\n", b"\n")
    return cleaned


def try_repair_a3m_file(path: str) -> tuple[bool, str]:
    target = Path(path)
    if not target.exists():
        return False, "file does not exist"
    try:
        original = target.read_bytes()
    except Exception as exc:
        return False, f"read failed: {exc}"
    cleaned = sanitize_a3m_bytes(original)
    valid, reason = validate_a3m_bytes(cleaned)
    if not valid:
        return False, reason
    if cleaned == original:
        return True, ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=str(target.parent),
            prefix=f"{target.name}.repair.",
        ) as handle:
            handle.write(cleaned)
            temp_path = Path(handle.name)
        temp_path.replace(target)
        return True, ""
    except Exception as exc:
        return False, f"repair write failed: {exc}"


def quarantine_invalid_msa(path: str) -> Optional[str]:
    src = Path(path)
    if not src.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantined = src.with_name(f"{src.stem}.invalid_{stamp}{src.suffix}")
    try:
        src.replace(quarantined)
        return str(quarantined)
    except Exception:
        return None


def save_generated_msa_to_repository(job_dir: str, destination_path: str) -> None:
    candidates = sorted(
        glob.glob(os.path.join(job_dir, "**", "*.a3m"), recursive=True),
        key=lambda p: ("processed" not in p, len(p)),
    )
    if not candidates:
        return
    src = None
    for candidate in candidates:
        valid, _ = is_valid_a3m_file(candidate)
        if not valid:
            repaired, _ = try_repair_a3m_file(candidate)
            if repaired:
                valid, _ = is_valid_a3m_file(candidate)
        if valid:
            src = candidate
            break
    if src is None:
        return
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination_parent = str(destination.parent)
    try:
        payload = sanitize_a3m_bytes(Path(src).read_bytes())
        valid_payload, _ = validate_a3m_bytes(payload)
        if not valid_payload:
            return
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=destination_parent,
            prefix=f"{destination.name}.tmp.",
        ) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        temp_path.replace(destination)
        valid_destination, _ = is_valid_a3m_file(str(destination))
        if not valid_destination and destination.exists():
            quarantine_invalid_msa(str(destination))
    except Exception:
        pass


def find_confidence_files(base_dir: str) -> dict:
    search_roots = [
        os.path.join(base_dir, "boltz_results_*", "predictions"),
        os.path.join(base_dir, "predictions"),
    ]
    pae = None
    plddt = None
    confidence_json = None
    json_files: list[str] = []
    for root_pattern in search_roots:
        for root in glob.glob(root_pattern):
            for path in glob.glob(os.path.join(root, "**", "*.npz"), recursive=True):
                name = os.path.basename(path).lower()
                if "pae" in name and pae is None:
                    pae = path
                if "plddt" in name and plddt is None:
                    plddt = path
            for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
                json_files.append(path)
                if confidence_json is None and os.path.basename(path).startswith("confidence"):
                    confidence_json = path
    if confidence_json is None and json_files:
        confidence_json = json_files[0]
    return {
        "pae": pae,
        "plddt": plddt,
        "confidence_json": confidence_json,
        "json_files": json_files,
    }


def collect_metrics(json_files: list[str]) -> dict:
    metrics = {}
    key_map = {
        "confidence": "confidence",
        "plddt": "plddt",
        "ptm": "ptm",
        "affinity": "affinity",
        "affinity_pred_value": "affinity",
        "affinity_probability_binary": "binding_probability",
    }
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        for source_key, target_key in key_map.items():
            if source_key in data and target_key not in metrics:
                metrics[target_key] = data[source_key]
    return metrics


def load_plddt(npz_path: Optional[str], json_path: Optional[str], structure_path: str) -> Optional[np.ndarray]:
    if structure_path.endswith(".cif"):
        cif_scores = extract_plddt_from_cif(structure_path)
        if cif_scores is not None and len(cif_scores):
            return cif_scores
    if npz_path and os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            for key in ("plddt", "predicted_lddt", "confidence", "data"):
                if key in data.files:
                    return squeeze_array(data[key])
            return squeeze_array(data[data.files[0]])
        except Exception:
            pass
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            for key in ("plddt", "atom_plddt", "confidence", "predicted_lddt"):
                if key in data:
                    arr = np.array(data[key])
                    return arr * 100 if arr.max() <= 1.0 else arr
        except Exception:
            pass
    return None


def load_pae(npz_path: Optional[str], json_path: Optional[str]) -> Optional[np.ndarray]:
    if npz_path and os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            for key in ("pae", "predicted_aligned_error", "data"):
                if key in data.files:
                    arr = data[key]
                    if arr.ndim == 3:
                        arr = arr[0]
                    if arr.ndim == 2:
                        return arr
        except Exception:
            pass
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            for key in ("pae", "predicted_aligned_error", "pae_matrix"):
                if key in data:
                    arr = np.array(data[key])
                    if arr.ndim == 3:
                        arr = arr[0]
                    if arr.ndim == 2:
                        return arr
        except Exception:
            pass
    return None


def squeeze_array(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr.mean(axis=-1) if arr.shape[-1] < arr.shape[0] else arr.mean(axis=0)
    arr = arr.flatten()
    if arr.max() <= 1.0:
        arr = arr * 100
    return arr


def extract_plddt_from_cif(cif_path: str) -> Optional[np.ndarray]:
    by_residue = {}
    try:
        with open(cif_path, "r", encoding="utf-8") as handle:
            headers: list[str] = []
            b_index = -1
            residue_index = -1
            in_atom_loop = False
            for raw_line in handle:
                line = raw_line.strip()
                if line.startswith("_atom_site."):
                    in_atom_loop = True
                    header = line.split(".", 1)[1]
                    headers.append(header)
                    if "B_iso" in header or "b_factor" in header.lower():
                        b_index = len(headers) - 1
                    if "label_seq_id" in header:
                        residue_index = len(headers) - 1
                    continue
                if in_atom_loop and line and not line.startswith(("_", "#")):
                    if line.startswith("loop_"):
                        in_atom_loop = False
                        continue
                    parts = line.split()
                    if min(b_index, residue_index) < 0 or len(parts) <= max(b_index, residue_index):
                        continue
                    try:
                        residue_id = int(parts[residue_index])
                        by_residue.setdefault(residue_id, float(parts[b_index]))
                    except (TypeError, ValueError):
                        continue
    except Exception:
        return None
    if not by_residue:
        return None
    return np.array([by_residue[idx] for idx in sorted(by_residue)])


def extract_error(output: str) -> str:
    if "CUDA out of memory" in output or "OutOfMemoryError" in output:
        return "GPU out of memory. Reduce sequence length or use fewer diffusion steps."
    if "docker: Error response from daemon" in output:
        return output.strip().splitlines()[-1]
    if "Traceback" in output:
        lines = [line for line in output.splitlines() if line.strip()]
        return "\n".join(lines[-15:])
    return output.strip() or "Prediction failed."
