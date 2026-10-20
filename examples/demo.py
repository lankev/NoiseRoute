"""Local interactive comparison; run with streamlit run examples/demo.py."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="NoiseRoute", page_icon="🔬", layout="wide")
st.title("NoiseRoute · placement et routage quantique")
st.caption(
    "Comparaison calculée sur votre machine. ESP = indicateur estimé, pas une fidélité physique mesurée."
)
with st.sidebar:
    family = st.selectbox("Circuit", ["random", "qft", "grover", "ansatz", "arithmetic"])
    n = st.slider("Qubits logiques", 3, 12, 5)
    topology = st.selectbox("Topologie", ["line", "ring", "grid", "heavy_hex", "neutral"])
    seed = st.number_input("Seed", 0, 1000000, 0)
    checkpoint = st.text_input("Checkpoint .pt ou .onnx (facultatif)", "")
    upload = st.file_uploader("Ou votre circuit OpenQASM 2", type=["qasm"])
    go = st.button("Compiler et comparer", type="primary")
st.info(
    "Sans checkpoint, NoiseRoute utilise son heuristique. Les modèles « smoke » fournis servent à tester le pipeline ; ils ne sont pas entraînés pour battre SABRE."
)
if go:
    try:
        if checkpoint and not Path(checkpoint).is_file():
            raise ValueError("Checkpoint introuvable.")
        config = dict(
            family=family,
            n=n,
            seed=int(seed),
            topology=topology,
            checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None,
            qasm=upload.getvalue().decode() if upload else None,
        )
        with (
            st.spinner("Compilation en cours…"),
            tempfile.TemporaryDirectory(prefix="noiseroute-demo-") as temp,
        ):
            temp = Path(temp)
            (temp / "input.json").write_text(json.dumps(config))
            completed = subprocess.run(
                [sys.executable, "-m", "noiseroute.demo_worker", str(temp)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if completed.returncode:
                raise RuntimeError(completed.stderr[-1500:])
            result = json.loads((temp / "output.json").read_text())
        st.write(
            f"{result['logical']} qubits logiques · {result['physical']} physiques · {result['edges']} arêtes"
        )
        rows = []
        for col, item in zip(st.columns(3), result["rows"]):
            with col:
                algorithm, metrics = item["algorithm"], item["metrics"]
                st.subheader(
                    {
                        "sabre": "SABRE",
                        "noise_sabre": "SABRE + layout bruit",
                        "learned": "NoiseRoute · modèle",
                        "noiseroute": "NoiseRoute · heuristique",
                    }[algorithm]
                )
                st.metric("ESP", f"{metrics['esp']:.4f}")
                st.metric("SWAP ajoutés", metrics["swaps_added"])
                st.metric("Compilation", f"{metrics['compile_ms']:.1f} ms")
                st.metric("Profondeur 2Q", metrics["depth_2q"])
                st.code(item["drawing"])
                rows.append(dict(algorithm=algorithm, **metrics))
        table = pd.DataFrame(rows)
        st.download_button(
            "Télécharger les mesures CSV", table.to_csv(index=False), "noiseroute-demo.csv", "text/csv"
        )
        st.caption(
            "Un circuit et une seed : démonstration, pas benchmark scientifique. Les calibrations sont synthétiques."
        )
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
