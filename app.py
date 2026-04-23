import os
import shutil
import signal
import threading
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from boltz_runner import (
    DEFAULT_BOLTZ_CACHE_DIR,
    DEFAULT_MSA_REPOSITORY_DIR,
    build_boltz_yaml_text,
    get_cached_msa_paths,
    is_valid_a3m_file,
    parse_fasta,
    run_prediction,
    validate_protein,
    validate_smiles,
)
from visualization import plot_pae, plot_plddt, viewer_html


EXAMPLE_PROTEIN = """>THRbeta_human
HKPEPTDEEWELIKTVTEAHVATNAQGSHWKQKRKFLPEDIGQAPIVNAPEGGKVDLEAFSHFTKIITPAITRVVDFAKKLPMFCELPCEDQIILLKGCCMEIMSLRAAVRYDPESETLTLNGEMAVTRGQLKNGGLGVVSDAIFDLGMSLSSFNLDDTEVALLQAVLLMSSDRPGLACVERIEKYQDSFLLAFEHYINYRKHHVTHFWPKLLMKVTDLRMIGACHASRFLHMKVECPTELFPPLFLEVFED"""

EXAMPLE_LIGAND = "OC1=C(I)C=C(OC2=C(I)C=C(C[C@H](N)C(O)=O)C=C2I)C=C1"


st.set_page_config(page_title="Boltz-2 Local", page_icon="🧬", layout="wide")
st.title("Boltz-2 Local")
st.caption("Streamlit host UI that runs Boltz-2 through a Docker container.")


def reset_prediction_state() -> None:
    st.session_state.pop("last_result", None)
    st.session_state["job_name"] = ""
    st.session_state["protein_input"] = ""
    st.session_state["ligand_input"] = ""


with st.sidebar:
    st.subheader("Runtime")
    gpu_device = st.selectbox(
        "GPU device",
        options=["0", "1", "all"],
        index=0,
        help="Select which host GPU Docker should expose to Boltz-2.",
    )
    with st.expander("Settings", expanded=False):
        docker_image = st.text_input(
            "Docker image",
            value=st.session_state.get("docker_image", "ovoex-boltz2"),
            key="docker_image",
        )
        cache_dir = st.text_input(
            "Cache directory",
            value=st.session_state.get("cache_dir", DEFAULT_BOLTZ_CACHE_DIR),
            key="cache_dir",
        )
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
        docker_args = st.text_input(
            "Docker extra args",
            value=st.session_state.get("docker_args", "--ipc=host --shm-size=48G"),
            key="docker_args",
        )
    st.code(
        f"BOLTZ_DOCKER_IMAGE={docker_image}\n"
        f"BOLTZ_CACHE_DIR={cache_dir}",
        language="bash",
    )
    with st.expander("Boltz settings", expanded=False):
        use_msa_server = st.checkbox("Use MSA", value=True)
        use_msa_repository = st.checkbox(
            "Use local MSA repository",
            value=True,
            help="Checks sequence-hash cache before contacting MSA server.",
        )
        use_potentials = st.checkbox(
            "Respect physics (use potentials)",
            value=True,
            help="Adds --use_potentials to improve physical plausibility.",
        )
        enable_affinity = st.checkbox(
            "Enable affinity for ligand runs",
            value=True,
            help="Injects properties.affinity into YAML when a ligand is present.",
        )
        sampling_steps = st.number_input(
            "Sampling steps",
            min_value=10,
            max_value=400,
            value=200,
            step=10,
            help="Boltz default: 200",
        )
        recycling_steps = st.number_input(
            "Recycling steps",
            min_value=1,
            max_value=12,
            value=3,
            step=1,
            help="Boltz default: 3",
        )
        diffusion_samples = st.number_input(
            "Diffusion samples",
            min_value=1,
            max_value=16,
            value=1,
            step=1,
            help="Boltz default: 1",
        )
        sampling_steps_affinity = st.number_input(
            "Affinity sampling steps",
            min_value=10,
            max_value=400,
            value=200,
            step=10,
            help="Boltz default: 200",
        )
        diffusion_samples_affinity = st.number_input(
            "Affinity diffusion samples",
            min_value=1,
            max_value=16,
            value=5,
            step=1,
            help="Boltz default: 5",
        )
        affinity_mw_correction = st.checkbox(
            "Affinity molecular-weight correction",
            value=False,
            help="Boltz default: off",
        )
    num_copies = st.slider("Copies", min_value=1, max_value=8, value=1, step=1)
    cyclic = st.checkbox("Cyclic peptide", value=False)
    if st.button("Load example"):
        st.session_state["protein_input"] = EXAMPLE_PROTEIN
        st.session_state["ligand_input"] = EXAMPLE_LIGAND
    if st.button("Stop application", type="secondary", use_container_width=True):
        st.warning("Stopping Streamlit. This browser tab will disconnect and the port will be freed.")
        threading.Timer(0.75, lambda: os.kill(os.getpid(), signal.SIGINT)).start()

left, right = st.columns([1, 1])
with left:
    job_name = st.text_input(
        "Job name",
        key="job_name",
        placeholder="example: insulin_test_01",
        help="Used as part of the results folder name.",
    )
    protein_text = st.text_area(
        "Protein sequence",
        key="protein_input",
        height=320,
        placeholder="Paste FASTA or raw amino acid sequence",
    )
    ligand_smiles = st.text_input(
        "Ligand SMILES (optional)",
        key="ligand_input",
        placeholder="CC(=O)Oc1ccccc1C(=O)O",
    )
    preview_text = (protein_text or "").strip()
    ok_sequence = False
    ok_smiles = False
    sequence_error = ""
    smiles_error = ""
    cleaned_sequence = ""
    cleaned_smiles = ""
    cached_msa_host = ""
    cached_msa_container = ""
    yaml_msa_path = None
    msa_cache_invalid_reason = ""

    if preview_text:
        _, parsed_sequence = parse_fasta(preview_text)
        ok_sequence, cleaned_sequence_or_error = validate_protein(parsed_sequence)
        ok_smiles, cleaned_smiles_or_error = validate_smiles(ligand_smiles or "")
        if not ok_sequence:
            sequence_error = cleaned_sequence_or_error
        if not ok_smiles:
            smiles_error = cleaned_smiles_or_error
        if ok_sequence:
            cleaned_sequence = cleaned_sequence_or_error
            cached_msa_host, cached_msa_container = get_cached_msa_paths(
                sequence=cleaned_sequence,
                msa_repository_dir=msa_repository_dir,
            )
            if use_msa_repository and os.path.exists(cached_msa_host):
                valid_msa, validation_reason = is_valid_a3m_file(cached_msa_host)
                if valid_msa:
                    yaml_msa_path = cached_msa_container
                else:
                    msa_cache_invalid_reason = validation_reason
        if ok_smiles:
            cleaned_smiles = cleaned_smiles_or_error or ""

    if use_msa_repository and preview_text:
        if not ok_sequence:
            st.caption("MSA cache status: sequence needs to be valid first.")
        elif yaml_msa_path:
            st.success(f"MSA available in repository: {cached_msa_host}")
        elif msa_cache_invalid_reason:
            st.error(
                "MSA cache file exists but is invalid and will be ignored: "
                f"{cached_msa_host} ({msa_cache_invalid_reason})"
            )
        else:
            st.warning("MSA not cached yet for this sequence. Server will be used once and then cached.")

    with st.popover("Preview YAML", use_container_width=True):
        if not preview_text:
            st.caption("Enter a protein sequence to preview YAML.")
        else:
            if not ok_sequence:
                st.error(sequence_error)
            elif not ok_smiles:
                st.error(smiles_error)
            else:
                yaml_preview = build_boltz_yaml_text(
                    sequence=cleaned_sequence,
                    ligand_smiles=cleaned_smiles or None,
                    msa_path=yaml_msa_path,
                    enable_affinity=enable_affinity,
                    num_copies=int(num_copies),
                    cyclic=cyclic,
                )
                if yaml_msa_path:
                    st.caption(f"MSA cache hit: {cached_msa_host}")
                elif use_msa_repository:
                    st.caption("MSA cache miss: server will be used and cache updated.")
                st.code(yaml_preview, language="yaml")
    action_left, action_right = st.columns(2)
    with action_left:
        run_clicked = st.button("Predict structure", type="primary", use_container_width=True)
    with action_right:
        st.button(
            "New prediction",
            use_container_width=True,
            on_click=reset_prediction_state,
        )
    if "last_result" not in st.session_state:
        st.caption("Fill inputs, preview YAML if needed, then run prediction.")

if run_clicked:
    if not job_name or not job_name.strip():
        st.session_state["last_result"] = None
        st.error("Job name is required.")
    else:
        with st.spinner("Running Boltz-2 in Docker..."):
            result = run_prediction(
                protein_text,
                ligand_smiles,
                job_name=job_name,
                results_dir=results_dir,
                cache_dir=cache_dir,
                msa_repository_dir=msa_repository_dir,
                docker_image=docker_image,
                docker_args=docker_args,
                use_msa_repository=use_msa_repository,
                use_potentials=use_potentials,
                enable_affinity=enable_affinity,
                sampling_steps=int(sampling_steps),
                recycling_steps=int(recycling_steps),
                diffusion_samples=int(diffusion_samples),
                sampling_steps_affinity=int(sampling_steps_affinity),
                diffusion_samples_affinity=int(diffusion_samples_affinity),
                affinity_mw_correction=affinity_mw_correction,
                num_copies=num_copies,
                cyclic=cyclic,
                use_msa_server=use_msa_server,
                gpu_device=gpu_device,
            )
        st.session_state["last_result"] = result

result = st.session_state.get("last_result")

with right:
    if result and not result.success:
        st.error(result.message)
        if result.job_dir:
            st.code(f"Result folder: {result.job_dir}")
        if result.raw_log:
            st.text_area("Logs", result.raw_log, height=280)
    elif result:
        st.success(result.message)
        st.code(f"Result folder: {result.job_dir}")
        if result.metrics:
            metric_columns = st.columns(min(4, max(1, len(result.metrics))))
            for index, (key, value) in enumerate(result.metrics.items()):
                if isinstance(value, list):
                    value = float(np.mean(value))
                if key == "binding_probability":
                    display = f"{value:.2%}"
                elif isinstance(value, (int, float)):
                    display = f"{value:.2f}"
                else:
                    display = str(value)
                metric_columns[index % len(metric_columns)].metric(key, display)
        if result.structure_text:
            components.html(viewer_html(result.structure_text, result.structure_format or "cif"), height=520)
        if result.structure_path:
            structure_path = Path(result.structure_path)
            st.download_button(
                "Download structure",
                data=structure_path.read_bytes(),
                file_name=structure_path.name,
                mime="chemical/x-cif" if structure_path.suffix == ".cif" else "chemical/x-pdb",
                use_container_width=True,
            )
        results_archive = Path(shutil.make_archive(result.job_dir, "zip", root_dir=result.job_dir))
        st.download_button(
            "Download full results",
            data=results_archive.read_bytes(),
            file_name=results_archive.name,
            mime="application/zip",
            use_container_width=True,
        )
        tabs = st.tabs(["PAE", "pLDDT", "YAML", "Logs"])
        with tabs[0]:
            pae_png = plot_pae(result.pae)
            if pae_png:
                st.image(pae_png)
            else:
                st.caption("No PAE output found.")
        with tabs[1]:
            plddt_png = plot_plddt(result.plddt)
            if plddt_png:
                st.image(plddt_png)
            else:
                st.caption("No pLDDT output found.")
        with tabs[2]:
            yaml_path = Path(result.job_dir) / "input.yaml"
            if yaml_path.exists():
                yaml_text = yaml_path.read_text(encoding="utf-8")
                st.code(yaml_text, language="yaml")
                st.download_button(
                    "Download input YAML",
                    data=yaml_text,
                    file_name=yaml_path.name,
                    mime="text/yaml",
                    use_container_width=True,
                )
            else:
                st.caption("No input.yaml found for this run.")
        with tabs[3]:
            st.text_area("Docker/Boltz logs", result.raw_log, height=320)
