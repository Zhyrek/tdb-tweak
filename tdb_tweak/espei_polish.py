"""Short ESPEI MCMC "polish" from the current parameters, run with the user's own ESPEI installation.

tdb-tweak does not depend on ESPEI: the run uses a Python interpreter that has ESPEI installed (for example the
espei-accel environment with its fast ``backend: cg``).  A small runner script executed by that interpreter:

1. reads an existing ESPEI input YAML as a template (its tags, prior, data weights, ... are kept), if given;
2. sets the phase models, datasets, ``mcmc.input_db`` (the current, edited TDB), iterations, chains,
   ``mcmc.symbols`` (the unlocked VV parameters) and the output files;
3. runs ESPEI.

The best parameters found (ESPEI's output TDB) are then applied as one undoable step."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Dict, List, Optional

import numpy as np
from PyQt6.QtCore import QProcess, QSettings, Qt, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget)

RUNNER = r'''
import json, os, sys
import yaml
cfg = json.load(open(sys.argv[1]))
base = {}
if cfg.get("template"):
    tdir = os.path.dirname(os.path.abspath(cfg["template"]))
    base = yaml.safe_load(open(cfg["template"])) or {}
    sysd = base.get("system", {})
    for k in ("phase_models", "datasets"):
        if sysd.get(k) and not os.path.isabs(sysd[k]):
            sysd[k] = os.path.join(tdir, sysd[k])
base.pop("generate_parameters", None)
sysd = base.setdefault("system", {})
if cfg.get("phase_models"):
    sysd["phase_models"] = cfg["phase_models"]
if cfg.get("datasets"):
    sysd["datasets"] = cfg["datasets"]
base["output"] = {"output_db": "polished.tdb", "tracefile": "trace.npy", "probfile": "lnprob.npy",
                  "verbosity": 1, "logfile": None}
m = base.setdefault("mcmc", {})
for k in ("restart_trace",):
    m.pop(k, None)
m.update(cfg["mcmc"])
m.setdefault("scheduler", None)
m.setdefault("save_interval", 1)
yaml.safe_dump(base, open("run.yaml", "w"))
print("ESPEI input written to run.yaml", flush=True)
sys.argv = ["espei", "--input", "run.yaml"]
from espei.espei_script import main
main()
'''


def parse_extra(text: str) -> Dict:
    """``key: value`` lines (values as JSON when possible, e.g. ``backend: cg``, ``cores: 4``)."""
    out = {}
    for ln in text.splitlines():
        ln = ln.split("#", 1)[0].strip()
        if not ln or ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        v = v.strip()
        try:
            out[k.strip()] = json.loads(v)
        except ValueError:
            out[k.strip()] = v
    return out


class EspeiDialog(QDialog):
    """Settings, live log and result of an ESPEI MCMC polish run."""
    resultReady = pyqtSignal(dict)            # VV name -> value of the best parameters

    def __init__(self, tdb_text: str, symbols: List[str], default_datasets: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("ESPEI MCMC polish")
        self.resize(820, 640)
        self.tdb_text, self.symbols = tdb_text, list(symbols)
        st = QSettings("tdb-tweak", "tdb-tweak")
        self.python = QLineEdit(st.value("espei/python", ""))
        self.python.setPlaceholderText("python with ESPEI installed, e.g. …/espei-accel/.venv/bin/python")
        self.template = QLineEdit(st.value("espei/template", ""))
        self.template.setPlaceholderText("optional: an ESPEI input YAML whose settings (tags, prior, …) are kept")
        self.phase_models = QLineEdit(st.value("espei/phase_models", ""))
        self.phase_models.setPlaceholderText("phase_models.json (taken from the template if empty)")
        self.datasets = QLineEdit(st.value("espei/datasets", "") or default_datasets)
        self.datasets.setPlaceholderText("datasets folder (taken from the template if empty)")
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 100000)
        self.iterations.setValue(int(st.value("espei/iterations", 50)))
        self.chains = QSpinBox()
        self.chains.setRange(2, 100)
        self.chains.setSingleStep(2)
        self.chains.setValue(int(st.value("espei/chains", 2)))
        self.std = QDoubleSpinBox()
        self.std.setRange(0.0001, 10)
        self.std.setDecimals(4)
        self.std.setValue(float(st.value("espei/std", 0.05)))
        self.extra = QPlainTextEdit(st.value("espei/extra", ""))
        self.extra.setPlaceholderText("extra mcmc settings, one per line, e.g.\nbackend: cg\ncores: 4")
        self.extra.setMaximumHeight(70)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.status = QLabel(f"{len(self.symbols)} unlocked VV parameters will be fitted (locked ones are kept).")
        self.status.setWordWrap(True)

        def browse(edit, title, flt="", folder=False):
            b = QPushButton("…")
            b.setMaximumWidth(30)

            def go():
                if folder:
                    p = QFileDialog.getExistingDirectory(self, title, edit.text())
                else:
                    p, _ = QFileDialog.getOpenFileName(self, title, edit.text(), flt)
                if p:
                    edit.setText(p)
            b.clicked.connect(go)
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(edit, 1)
            h.addWidget(b)
            return w

        form = QFormLayout()
        form.addRow("ESPEI Python", browse(self.python, "Python with ESPEI"))
        form.addRow("Template YAML", browse(self.template, "ESPEI input YAML", "YAML (*.yaml *.yml)"))
        form.addRow("Phase models", browse(self.phase_models, "phase_models.json", "JSON (*.json)"))
        form.addRow("Datasets", browse(self.datasets, "Datasets folder", folder=True))
        row = QHBoxLayout()
        row.addWidget(QLabel("Iterations"))
        row.addWidget(self.iterations)
        row.addWidget(QLabel("Chains per parameter"))
        row.addWidget(self.chains)
        row.addWidget(QLabel("Chain std. deviation (fraction)"))
        row.addWidget(self.std)
        form.addRow(row)
        form.addRow("Extra mcmc", self.extra)
        self.run_btn = QPushButton("Run")
        self.run_btn.setStyleSheet("font-weight:bold;")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run)
        self.stop_btn.clicked.connect(self.stop)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.close)
        btns = QHBoxLayout()
        btns.addWidget(self.run_btn)
        btns.addWidget(self.stop_btn)
        btns.addStretch(1)
        btns.addWidget(bb)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.status)
        lay.addWidget(self.log, 1)
        lay.addLayout(btns)
        self.proc: Optional[QProcess] = None
        self.workdir: Optional[str] = None

    def _remember(self):
        st = QSettings("tdb-tweak", "tdb-tweak")
        for k, w in (("python", self.python), ("template", self.template), ("phase_models", self.phase_models),
                     ("datasets", self.datasets)):
            st.setValue(f"espei/{k}", w.text().strip())
        st.setValue("espei/iterations", self.iterations.value())
        st.setValue("espei/chains", self.chains.value())
        st.setValue("espei/std", self.std.value())
        st.setValue("espei/extra", self.extra.toPlainText())

    def config(self, workdir: str) -> Dict:
        mcmc = {"input_db": os.path.join(workdir, "start.tdb"), "iterations": self.iterations.value(),
                "chains_per_parameter": self.chains.value(), "chain_std_deviation": self.std.value(),
                "symbols": self.symbols}
        mcmc.update(parse_extra(self.extra.toPlainText()))
        return {"template": self.template.text().strip() or None,
                "phase_models": self.phase_models.text().strip() or None,
                "datasets": self.datasets.text().strip() or None, "mcmc": mcmc}

    def run(self):
        py = self.python.text().strip() or sys.executable
        if not self.symbols:
            self.status.setText("<span style='color:#b00'>No unlocked VV parameter to fit.</span>")
            return
        if not (self.template.text().strip() or (self.phase_models.text().strip() and self.datasets.text().strip())):
            self.status.setText("<span style='color:#b00'>Give a template YAML, or phase models and a datasets "
                                "folder.</span>")
            return
        self._remember()
        self.workdir = tempfile.mkdtemp(prefix="tdbtweak-espei-")
        with open(os.path.join(self.workdir, "start.tdb"), "w", encoding="utf-8") as f:
            f.write(self.tdb_text)
        with open(os.path.join(self.workdir, "settings.json"), "w") as f:
            json.dump(self.config(self.workdir), f, indent=1)
        with open(os.path.join(self.workdir, "runner.py"), "w") as f:
            f.write(RUNNER)
        self.log.clear()
        self.log.appendPlainText(f"working folder: {self.workdir}\n{py} runner.py settings.json\n")
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(self.workdir)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(
            lambda: self.log.appendPlainText(bytes(self.proc.readAllStandardOutput()).decode(errors="replace").rstrip()))
        self.proc.finished.connect(self._finished)
        self.proc.start(py, ["runner.py", "settings.json"])
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText(f"running ESPEI MCMC on {len(self.symbols)} parameters…")

    def stop(self):
        if self.proc is not None:
            self.proc.kill()

    def _finished(self, code, status):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        out = os.path.join(self.workdir or "", "polished.tdb")
        if code != 0 or not os.path.exists(out):
            self.status.setText(f"<span style='color:#b00'>ESPEI stopped (exit code {code}); see the log.</span>")
            return
        from .tdbmodel import TDBDocument
        best = TDBDocument.from_file(out).vv_values
        vals = {k: v for k, v in best.items() if k in self.symbols}
        msg = f"done: {len(vals)} parameters from the best sample."
        try:
            lnp = np.load(os.path.join(self.workdir, "lnprob.npy"))
            first = float(np.nanmax(lnp[:, 0])) if lnp.ndim == 2 else float("nan")
            msg += f"  ln p: start {first:.4g} → best {float(np.nanmax(lnp)):.4g}."
        except Exception:
            pass
        self.status.setText(msg + " Applied as one undo step (Ctrl+Z reverts).")
        self.resultReady.emit(vals)
