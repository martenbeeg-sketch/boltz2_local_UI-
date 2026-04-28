import base64
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def viewer_html(structure_text: str, fmt: str = "cif") -> str:
    escaped = (
        structure_text.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("$", "\\$")
        .replace("\r\n", "\\n")
        .replace("\r", "\\n")
        .replace("\n", "\\n")
        .replace("'", "\\'")
        .replace('"', '\\"')
    )
    return f"""<!DOCTYPE html>
<html>
<head>
  <script src="https://3dmol.org/build/3Dmol-min.js"></script>
  <style>
    body {{ margin: 0; background: #0f172a; }}
    #viewer {{ width: 100%; height: 500px; }}
  </style>
</head>
<body>
  <div id="viewer"></div>
  <script>
    const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "#0f172a" }});
    let structure = "{escaped}";
    structure = structure.replace(/\\\\n/g, "\\n");
    const model = viewer.addModel(structure, "{fmt}");
    viewer.setStyle({{}}, {{
      cartoon: {{
        colorfunc: function(atom) {{
          const b = atom.b || 0;
          if (b > 90) return "#0053d6";
          if (b > 70) return "#65cbf3";
          if (b > 50) return "#ffdb13";
          return "#ff7d45";
        }}
      }}
    }});
    // Keep original hetero styling for small molecules.
    viewer.addStyle({{hetflag: true}}, {{
      stick: {{ colorscheme: "greenCarbon", radius: 0.2 }},
      sphere: {{ scale: 0.25 }}
    }});
    // Explicit ion overlay so monoatomic ions remain visible (e.g., Na+).
    // Detect ions from parsed atoms to avoid selector mismatches across formats.
    const ionTags = new Set(["NA", "K", "CA", "MG", "ZN", "MN", "FE", "CU", "CL"]);
    const hetAtoms = model.selectedAtoms({{ hetflag: true }}) || [];
    const ionSerials = [];
    for (const atom of hetAtoms) {{
      const elem = String(atom.elem || "").toUpperCase();
      const atomName = String(atom.atom || "").toUpperCase();
      const resn = String(atom.resn || "").toUpperCase();
      if (ionTags.has(elem) || ionTags.has(atomName) || ionTags.has(resn)) {{
        if (atom.serial !== undefined) {{
          ionSerials.push(atom.serial);
        }}
      }}
    }}
    if (ionSerials.length > 0) {{
      viewer.setStyle({{ serial: ionSerials }}, {{
        sphere: {{ colorscheme: "Jmol", radius: 1.45 }}
      }});
    }}
    viewer.zoomTo();
    viewer.render();
  </script>
</body>
</html>"""


def plot_plddt(scores: np.ndarray) -> Optional[bytes]:
    if scores is None or not len(scores):
        return None
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(1, len(scores) + 1)
    ax.plot(x, scores, color="#0053d6", linewidth=1.4)
    ax.fill_between(x, scores, color="#65cbf3", alpha=0.25)
    ax.set(xlabel="Residue", ylabel="pLDDT", ylim=(0, 100), xlim=(1, len(scores)))
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    return _fig_to_png(fig)


def plot_pae(matrix: np.ndarray) -> Optional[bytes]:
    if matrix is None or matrix.ndim != 2:
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Greens_r", vmin=0, vmax=30, aspect="equal")
    fig.colorbar(image, ax=ax, shrink=0.8, pad=0.02, label="PAE (A)")
    ax.set(xlabel="Scored residue", ylabel="Aligned residue")
    fig.tight_layout()
    return _fig_to_png(fig)


def png_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _fig_to_png(fig) -> bytes:
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
