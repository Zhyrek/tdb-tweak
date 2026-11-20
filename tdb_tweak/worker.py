"""The mapping worker: one QThread that owns the Engine (and its compiled models) for the whole session."""
from __future__ import annotations

import threading
import traceback
from typing import Dict, Optional, Sequence

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from .engine import Cancelled, Engine, MapRequest, MapResult
from .optimize import optimize_invariant


class _Worker(QObject):
    progress = pyqtSignal(int, object)
    finished = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    cancelled = pyqtSignal(int)
    status = pyqtSignal(int, str)
    opt_progress = pyqtSignal(int, object)
    opt_finished = pyqtSignal(int, object)
    probe_ready = pyqtSignal(object, object)
    scan_ready = pyqtSignal(object)
    task_ready = pyqtSignal(int, object)

    def __init__(self):
        super().__init__()
        self.engine: Optional[Engine] = None
        self.cancel_events: Dict[int, threading.Event] = {}

    @pyqtSlot(int, str, object, object, object, object, bool, object)
    def run(self, job: int, compute_text: str, param_names, values, request, zpf, warm, thermo):
        ev = self.cancel_events.get(job)
        if ev is not None and ev.is_set():
            self.cancelled.emit(job)
            return
        try:
            if self.engine is None:
                self.engine = Engine()
            if self.engine.set_database(compute_text, list(param_names)):
                self.status.emit(job, "database structure changed: models will be compiled on this run")
            res = self.engine.run(request, values, cancel=ev,
                                  progress=lambda r: self.progress.emit(job, r), warm=warm)
            if zpf:
                try:
                    from .zpfscore import zpf_score
                    res.zpf = zpf_score(self.engine, request, values, zpf)
                except Exception as ex:
                    res.message = (res.message + "; " if res.message else "") + f"ZPF score: {ex}"
            if thermo:
                try:
                    from .thermo import thermo_score
                    res.thermo = thermo_score(self.engine, request, values, thermo)
                except Exception as ex:
                    res.message = (res.message + "; " if res.message else "") + f"thermo score: {ex}"
            self.finished.emit(job, res)
        except Cancelled:
            self.cancelled.emit(job)
        except Exception as ex:
            self.failed.emit(job, f"{type(ex).__name__}: {ex}\n\n{traceback.format_exc()}")
        finally:
            self.cancel_events.pop(job, None)

    @pyqtSlot(str, object, object, object, object)
    def run_probe(self, compute_text: str, param_names, values, request, point):
        """One equilibrium; queued behind any running job (it never cancels one)."""
        try:
            if self.engine is None:
                self.engine = Engine()
            self.engine.set_database(compute_text, list(param_names))
            T, x = point
            self.probe_ready.emit(point, self.engine.point(request, values, T, x))
        except Exception as ex:
            self.probe_ready.emit(point, f"{type(ex).__name__}: {ex}")

    @pyqtSlot(int, str, str, object, object, object, object)
    def run_task(self, tid: int, kind: str, compute_text: str, param_names, values, request, args):
        """Small computations for companion windows (queued behind maps, never cancelling them)."""
        try:
            if self.engine is None:
                self.engine = Engine()
            self.engine.set_database(compute_text, list(param_names))
            if kind == "gx":
                out = self.engine.gx_curves(request, values, args["T"])
            elif kind == "step":
                out = self.engine.step(request, values, args["comp"], args["tmin"], args["tmax"])
            else:
                raise ValueError(kind)
            self.task_ready.emit(tid, out)
        except Exception as ex:
            self.task_ready.emit(tid, f"{type(ex).__name__}: {ex}")

    @pyqtSlot(str, object, object, object)
    def run_scan(self, compute_text: str, param_names, values, request):
        try:
            if self.engine is None:
                self.engine = Engine()
            self.engine.set_database(compute_text, list(param_names))
            from .artifacts import scan
            self.scan_ready.emit(scan(self.engine, request.elements, values, phases=request.phases))
        except Exception as ex:
            self.scan_ready.emit(f"{type(ex).__name__}: {ex}")

    @pyqtSlot(int, str, object, object, object, object)
    def run_opt(self, job: int, compute_text: str, param_names, values, request, spec):
        ev = self.cancel_events.get(job)
        if ev is not None and ev.is_set():
            self.cancelled.emit(job)
            return
        try:
            if self.engine is None:
                self.engine = Engine()
            self.engine.set_database(compute_text, list(param_names))
            variables, target, settings = spec
            res = optimize_invariant(self.engine, request, values, variables, target, settings, cancel=ev,
                                     progress=lambda it: self.opt_progress.emit(job, it))
            # what else might the new parameters have broken?
            try:
                from .artifacts import scan
                final = dict(values)
                final.update(res.iterations[-1].values)
                systems = {}
                for t in res.targets:
                    systems.setdefault(tuple(t.req.elements), t.req)
                res.artifacts = []
                for els, rq in systems.items():
                    res.artifacts += scan(self.engine, els, final, phases=rq.phases)
            except Exception as ex:
                res.artifacts = []
                res.message += f" (artifact scan failed: {ex})"
            self.opt_finished.emit(job, res)
        except Cancelled:
            self.cancelled.emit(job)
        except Exception as ex:
            self.failed.emit(job, f"{type(ex).__name__}: {ex}\n\n{traceback.format_exc()}")
        finally:
            self.cancel_events.pop(job, None)


class MapRunner(QObject):
    """Queue map jobs; a new job cancels the running one."""
    progress = pyqtSignal(object)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    status = pyqtSignal(str)
    opt_progress = pyqtSignal(object)
    opt_finished = pyqtSignal(object)
    _submit = pyqtSignal(int, str, object, object, object, object, bool, object)
    _submit_opt = pyqtSignal(int, str, object, object, object, object)
    _submit_probe = pyqtSignal(str, object, object, object, object)
    probe_ready = pyqtSignal(object, object)
    _submit_scan = pyqtSignal(str, object, object, object)
    scan_ready = pyqtSignal(object)
    _submit_task = pyqtSignal(int, str, str, object, object, object, object)
    task_ready = pyqtSignal(int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = QThread()
        self._thread.setObjectName("map-worker")
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._submit.connect(self._worker.run)
        self._submit_opt.connect(self._worker.run_opt)
        self._submit_probe.connect(self._worker.run_probe)
        self._submit_scan.connect(self._worker.run_scan)
        self._submit_task.connect(self._worker.run_task)
        self._worker.task_ready.connect(self.task_ready)
        self._tid = 0
        self._worker.scan_ready.connect(self.scan_ready)
        self._worker.probe_ready.connect(self.probe_ready)
        self._worker.opt_progress.connect(lambda j, it: self.opt_progress.emit(it) if j == self._job else None)
        self._worker.opt_finished.connect(self._on_opt_finished)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.status.connect(lambda j, s: self.status.emit(s) if j == self._job else None)
        self._thread.start()
        self._job = 0
        self.running = False
        self.optimizing = False

    def submit(self, compute_text: str, param_names: Sequence[str], values: Dict[str, float], request: MapRequest,
               zpf=None, warm: bool = False, thermo=None):
        """Map ``request``; with ZPF datasets, the result also carries their ZPF score (``result.zpf``)."""
        self.cancel()
        self._job += 1
        self._worker.cancel_events[self._job] = threading.Event()
        self.running = True
        self._submit.emit(self._job, compute_text, list(param_names), dict(values), request,
                          [d for d in (zpf or []) if d.enabled], bool(warm),
                          [d for d in (thermo or []) if d.enabled])

    def submit_opt(self, compute_text: str, param_names: Sequence[str], values: Dict[str, float],
                   request: MapRequest, spec):
        """Run the invariant optimiser; ``spec`` = (variables, target, settings)."""
        self.cancel()
        self._job += 1
        self._worker.cancel_events[self._job] = threading.Event()
        self.running = True
        self.optimizing = True
        self._submit_opt.emit(self._job, compute_text, list(param_names), dict(values), request, spec)

    def submit_probe(self, compute_text: str, param_names, values, request, T: float, x: Dict[str, float]):
        self._submit_probe.emit(compute_text, list(param_names), dict(values), request, (T, dict(x)))

    def submit_task(self, kind: str, compute_text: str, param_names, values, request, args) -> int:
        self._tid += 1
        self._submit_task.emit(self._tid, kind, compute_text, list(param_names), dict(values), request, dict(args))
        return self._tid

    def submit_scan(self, compute_text: str, param_names, values, request):
        self._submit_scan.emit(compute_text, list(param_names), dict(values), request)

    def _on_opt_finished(self, job, res):
        if job == self._job:
            self.running = False
            self.optimizing = False
            self.opt_finished.emit(res)

    def cancel(self):
        for ev in list(self._worker.cancel_events.values()):
            ev.set()

    def _on_progress(self, job, res):
        if job == self._job:
            self.progress.emit(res)

    def _on_finished(self, job, res):
        if job == self._job:
            self.running = False
            self.finished.emit(res)

    def _on_failed(self, job, msg):
        if job == self._job:
            self.running = False
            self.optimizing = False
            self.failed.emit(msg)

    def _on_cancelled(self, job):
        if job == self._job:
            self.running = False
            self.optimizing = False
            self.cancelled.emit()

    def shutdown(self):
        self.cancel()
        self._thread.quit()
        self._thread.wait(5000)
