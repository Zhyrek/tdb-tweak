"""A pool of worker processes for mapping several diagrams at once ("Map all views").

Each process keeps its own Engine, so the compiled models of a database are reused by later jobs in that process;
only the first use pays the process start-up (importing pycalphad) and compilation."""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence

_engine = None


def _job(compute_text: str, param_names: Sequence[str], values: Dict[str, float], request, zpf):
    global _engine
    from .engine import Engine
    if _engine is None:
        _engine = Engine()
    _engine.set_database(compute_text, list(param_names))
    res = _engine.run(request, values)
    if zpf:
        try:
            from .zpfscore import zpf_score
            res.zpf = zpf_score(_engine, request, values, zpf)
        except Exception as ex:
            res.message = (res.message + "; " if res.message else "") + f"ZPF score: {ex}"
    return res


def _warm():
    from . import engine  # noqa: F401  (imports pycalphad in the worker)
    return os.getpid()


class MapPool:
    def __init__(self, workers: Optional[int] = None):
        n = workers or max(1, min(4, (os.cpu_count() or 2) - 1))
        self.workers = n
        self._ex: Optional[ProcessPoolExecutor] = None

    def _executor(self) -> ProcessPoolExecutor:
        if self._ex is None:
            self._ex = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"))
        return self._ex

    def warm_up(self):
        """Start the processes in the background (they import pycalphad once)."""
        ex = self._executor()
        for _ in range(self.workers):
            ex.submit(_warm)

    def submit(self, compute_text: str, param_names, values, request, zpf=None) -> Future:
        return self._executor().submit(_job, compute_text, list(param_names), dict(values), request,
                                       [d for d in (zpf or []) if d.enabled])

    def shutdown(self):
        if self._ex is not None:
            self._ex.shutdown(wait=False, cancel_futures=True)
            self._ex = None
