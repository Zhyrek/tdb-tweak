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

    def __init__(self):
        super().__init__()
        self.engine: Optional[Engine] = None
        self.cancel_events: Dict[int, threading.Event] = {}

    @pyqtSlot(int, str, object, object, object)
    def run(self, job: int, compute_text: str, param_names, values, request):
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
                                  progress=lambda r: self.progress.emit(job, r))
            self.finished.emit(job, res)
        except Cancelled:
            self.cancelled.emit(job)
        except Exception as ex:
            self.failed.emit(job, f"{type(ex).__name__}: {ex}\n\n{traceback.format_exc()}")
        finally:
            self.cancel_events.pop(job, None)

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
    _submit = pyqtSignal(int, str, object, object, object)
    _submit_opt = pyqtSignal(int, str, object, object, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = QThread()
        self._thread.setObjectName("map-worker")
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._submit.connect(self._worker.run)
        self._submit_opt.connect(self._worker.run_opt)
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

    def submit(self, compute_text: str, param_names: Sequence[str], values: Dict[str, float], request: MapRequest):
        self.cancel()
        self._job += 1
        self._worker.cancel_events[self._job] = threading.Event()
        self.running = True
        self._submit.emit(self._job, compute_text, list(param_names), dict(values), request)

    def submit_opt(self, compute_text: str, param_names: Sequence[str], values: Dict[str, float],
                   request: MapRequest, spec):
        """Run the invariant optimiser; ``spec`` = (variables, target, settings)."""
        self.cancel()
        self._job += 1
        self._worker.cancel_events[self._job] = threading.Event()
        self.running = True
        self.optimizing = True
        self._submit_opt.emit(self._job, compute_text, list(param_names), dict(values), request, spec)

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
