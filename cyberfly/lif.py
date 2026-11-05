import ctypes as C
import hashlib
import subprocess
from pathlib import Path

import numpy as np

PARAMS = dict(v_0=-52.0, v_rst=-52.0, v_th=-45.0, t_mbr=20.0, tau=5.0, t_rfc=2.2, t_dly=1.8, w_syn=0.275, dt=0.1)
RFC_STEPS = round(PARAMS["t_rfc"] / PARAMS["dt"])
DELAY_STEPS = round(PARAMS["t_dly"] / PARAMS["dt"])
KICK = 68.75

_SRC = Path(__file__).with_name("lif.c")


def _library() -> C.CDLL:
    digest = hashlib.sha1(_SRC.read_bytes()).hexdigest()[:12]
    out = Path.home() / ".cache" / "cyberfly" / f"lif-{digest}.so"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cc", "-O3", "-march=native", "-shared", "-fPIC", "-o", str(out), str(_SRC), "-lm"], check=True)
    lib = C.CDLL(str(out))
    lib.sim_new.restype = C.c_void_p
    lib.sim_new.argtypes = [C.c_int32, C.c_void_p, C.c_void_p, C.c_void_p] + [C.c_double] * 6 + [C.c_int32, C.c_int32, C.c_double]
    lib.sim_run.restype = C.c_int64
    lib.sim_run.argtypes = [C.c_void_p, C.c_int64, C.c_int64, C.c_void_p, C.c_void_p, C.c_void_p,
                            C.c_void_p, C.c_int64, C.c_void_p, C.c_void_p]
    lib.sim_time.restype = C.c_int64
    lib.sim_time.argtypes = [C.c_void_p]
    lib.sim_reset.argtypes = [C.c_void_p]
    lib.sim_voltage.argtypes = [C.c_void_p, C.c_int32, C.c_void_p, C.c_void_p]
    return lib


def csr(n: int, pre, post, w):
    order = np.lexsort((post, pre))
    pre, post, w = pre[order], post[order], w[order]
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, pre + 1, 1)
    return np.cumsum(indptr), post.astype(np.int32), w


class Sim:
    def __init__(self, n: int, pre, post, count, eps: float = 1e-6):
        self.lib = _library()
        self.n = n
        w = np.asarray(count, np.float64) * PARAMS["w_syn"]
        self.indptr, self.post, w = csr(n, np.asarray(pre, np.int64), np.asarray(post, np.int64), w)
        self.w = w.astype(np.float32)
        p = PARAMS
        self.h = self.lib.sim_new(n, self.indptr.ctypes.data, self.post.ctypes.data, self.w.ctypes.data,
                                  p["v_0"], p["v_rst"], p["v_th"], p["t_mbr"], p["tau"], p["dt"],
                                  DELAY_STEPS, RFC_STEPS, eps)

    @property
    def t(self) -> int:
        return self.lib.sim_time(self.h)

    def voltage(self, ids) -> np.ndarray:
        ids = np.ascontiguousarray(ids, np.int32)
        out = np.empty(len(ids), np.float64)
        self.lib.sim_voltage(self.h, len(ids), ids.ctypes.data, out.ctypes.data)
        return out

    def reset(self):
        self.lib.sim_reset(self.h)

    def run(self, steps: int, kicks=None, record=None, cap: int = 50_000_000):
        if kicks is None or not len(kicks[0]):
            kt, ki, ka = np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.float32)
        else:
            o = np.argsort(kicks[0], kind="stable")
            kt = np.ascontiguousarray(np.asarray(kicks[0])[o], np.int64)
            ki = np.ascontiguousarray(np.asarray(kicks[1])[o], np.int32)
            ka = np.ascontiguousarray(np.asarray(kicks[2])[o], np.float32)
        rp = None
        if record is not None:
            record = np.ascontiguousarray(record, np.uint8)
            rp = record.ctypes.data
        ot = np.empty(cap, np.int64)
        oi = np.empty(cap, np.int32)
        m = self.lib.sim_run(self.h, int(steps), len(kt), kt.ctypes.data, ki.ctypes.data, ka.ctypes.data,
                             rp, cap, ot.ctypes.data, oi.ctypes.data)
        if m > cap:
            raise RuntimeError(f"spike buffer overflow: {m} > {cap}")
        return ot[:m].copy(), oi[:m].copy()


def simulator(n: int, pre, post, count, gpu: bool = False):
    if gpu:
        from .lif_gpu import GPUSim
        return GPUSim(n, pre, post, count)
    return Sim(n, pre, post, count)
