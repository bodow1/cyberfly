import cupy as cp
import numpy as np

from .lif import DELAY_STEPS, PARAMS, RFC_STEPS, csr

_SRC = r"""
extern "C" {
__device__ __forceinline__ bool refr(long long t, const long long *last, const int *rfc, int i) {
  return t - last[i] < rfc[i];
}
__global__ void update(int n, const long long *T, double *v, double *g, const long long *last, const int *rfc,
                       double v0, double vth, double em, double es, double cc, int *spk, int *nsp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  long long t = *T;
  if (refr(t, last, rfc, i)) return;
  double u = v[i] - v0, gg = g[i];
  if (u == 0.0 && gg == 0.0) return;
  u = u * em + gg * cc; gg = gg * es;
  v[i] = u + v0; g[i] = gg;
  if (v[i] > vth) { int k = atomicAdd(nsp, 1); spk[k] = i; }
}
__global__ void deliver(const long long *T, int r, int n, const int *ring, const int *ring_n,
                        const long long *indptr, const int *post, const int *cnt, const long long *last,
                        const int *rfc, int *acc) {
  long long t = *T; int slot = (int)(t % r); int m = ring_n[slot];
  const int *sl = ring + (long long)slot * n;
  for (int s = blockIdx.x; s < m; s += gridDim.x) {
    int i = sl[s];
    for (long long e = indptr[i] + threadIdx.x; e < indptr[i + 1]; e += blockDim.x) {
      int j = post[e];
      if (refr(t, last, rfc, j)) continue;
      atomicAdd(&acc[j], cnt[e]);
    }
  }
}
__global__ void apply(int n, double *g, int *acc, double wsyn) {
  int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= n) return;
  int a = acc[j]; if (a != 0) { g[j] += (double)a * wsyn; acc[j] = 0; }
}
__global__ void kicks(const long long *T, long long t0, const int *kptr, int steps, double *v, const long long *last,
                      const int *rfc, const int *kid, const float *kamt) {
  long long t = *T; long long r = t - t0; if (r < 0 || r >= steps) return;
  for (int k = kptr[r] + threadIdx.x; k < kptr[r + 1]; k += blockDim.x) {
    int i = kid[k];
    if (!refr(t, last, rfc, i)) atomicAdd(&v[i], (double)kamt[k]);
  }
}
__global__ void reset(const long long *T, int r, int n, int dly, double *v, double *g, long long *last,
                      const int *spk, const int *nsp, double vrst, int *ring, const unsigned char *rec,
                      long long *ost, int *oid, int *on, long long cap) {
  long long t = *T; int m = *nsp; int oslot = (int)((t + dly) % r);
  int *ro = ring + (long long)oslot * n;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < m; k += gridDim.x * blockDim.x) {
    int i = spk[k];
    v[i] = vrst; g[i] = 0.0; last[i] = t; ro[k] = i;
    if (rec[i]) { int q = atomicAdd(on, 1); if (q < cap) { ost[q] = t; oid[q] = i; } }
  }
}
__global__ void advance(long long *T, int r, int dly, int *nsp, int *ring_n) {
  long long t = *T;
  ring_n[(int)((t + dly) % r)] = *nsp;
  ring_n[(int)(t % r)] = 0;
  *nsp = 0; *T = t + 1;
}
}
"""

_module = cp.RawModule(code=_SRC)
_update, _deliver, _apply, _kicks, _reset, _advance = (
    _module.get_function(n) for n in ("update", "deliver", "apply", "kicks", "reset", "advance"))


class GPUSim:
    BLOCK = 98

    def __init__(self, n: int, pre, post, count):
        self.n = n
        pre = np.asarray(pre, np.int64)
        post = np.asarray(post, np.int64)
        cnt = np.asarray(count, np.int64)
        order = np.lexsort((post, pre))
        indptr, post32, _ = csr(n, pre, post, cnt.astype(np.float64))
        self.indptr = cp.asarray(indptr)
        self.post = cp.asarray(post32)
        self.cnt = cp.asarray(cnt[order].astype(np.int32))
        self.v = cp.full(n, PARAMS["v_0"], cp.float64)
        self.g = cp.zeros(n, cp.float64)
        self.last = cp.full(n, -10**9, cp.int64)
        self.rfc = cp.full(n, RFC_STEPS, cp.int32)
        self.acc = cp.zeros(n, cp.int32)
        dt = PARAMS["dt"]
        self.em = np.exp(-dt / PARAMS["t_mbr"])
        self.es = np.exp(-dt / PARAMS["tau"])
        self.cc = PARAMS["tau"] / (PARAMS["tau"] - PARAMS["t_mbr"]) * (self.es - self.em)
        self.r = DELAY_STEPS + 1
        self.ring = cp.zeros(self.r * n, cp.int32)
        self.ring_n = cp.zeros(self.r, cp.int32)
        self.spk = cp.zeros(n, cp.int32)
        self.nsp = cp.zeros(1, cp.int32)
        self.T = cp.zeros(1, cp.int64)
        self.stream = cp.cuda.Stream(non_blocking=True)

    @property
    def t(self) -> int:
        return int(self.T.get()[0])

    def _step(self, t0: int, steps: int):
        n, b = self.n, 256
        grid = (n + b - 1) // b
        s = self.stream
        _update((grid,), (b,), (np.int32(n), self.T, self.v, self.g, self.last, self.rfc, np.float64(PARAMS["v_0"]),
                                np.float64(PARAMS["v_th"]), np.float64(self.em), np.float64(self.es),
                                np.float64(self.cc), self.spk, self.nsp), stream=s)
        _deliver((256,), (128,), (self.T, np.int32(self.r), np.int32(n), self.ring, self.ring_n, self.indptr,
                                  self.post, self.cnt, self.last, self.rfc, self.acc), stream=s)
        _apply((grid,), (b,), (np.int32(n), self.g, self.acc, np.float64(PARAMS["w_syn"])), stream=s)
        _kicks((1,), (256,), (self.T, np.int64(t0), self.kptr, np.int32(steps), self.v, self.last, self.rfc,
                              self.kid, self.kamt), stream=s)
        _reset((64,), (256,), (self.T, np.int32(self.r), np.int32(n), np.int32(DELAY_STEPS), self.v, self.g,
                               self.last, self.spk, self.nsp, np.float64(PARAMS["v_rst"]), self.ring, self.rec,
                               self.ost, self.oid, self.on, np.int64(self.cap)), stream=s)
        _advance((1,), (1,), (self.T, np.int32(self.r), np.int32(DELAY_STEPS), self.nsp, self.ring_n), stream=s)

    def run(self, steps: int, kicks=None, record=None, cap: int = 50_000_000):
        t0 = self.t
        ks = np.zeros(0, np.int64)
        kid = np.zeros(1, np.int32)
        kamt = np.zeros(1, np.float32)
        if kicks is not None and len(kicks[0]):
            o = np.argsort(kicks[0], kind="stable")
            ks = np.asarray(kicks[0], np.int64)[o]
            kid = np.asarray(kicks[1], np.int32)[o]
            kamt = np.asarray(kicks[2], np.float32)[o]
        self.kptr = cp.asarray(np.searchsorted(ks, np.arange(t0, t0 + steps + 1)).astype(np.int32))
        self.kid = cp.asarray(kid)
        self.kamt = cp.asarray(kamt)
        self.rec = cp.asarray(np.ones(self.n, np.uint8) if record is None else np.asarray(record, np.uint8))
        self.cap = int(min(cap, 150_000_000))
        self.ost = cp.zeros(self.cap, cp.int64)
        self.oid = cp.zeros(self.cap, cp.int32)
        self.on = cp.zeros(1, cp.int32)
        full, rest = divmod(steps, self.BLOCK)
        if full:
            with self.stream:
                self.stream.begin_capture()
                for _ in range(self.BLOCK):
                    self._step(t0, steps)
                graph = self.stream.end_capture()
            for _ in range(full):
                graph.launch(stream=self.stream)
        for _ in range(rest):
            self._step(t0, steps)
        self.stream.synchronize()
        m = int(self.on.get()[0])
        if m > self.cap:
            raise RuntimeError("spike buffer overflow")
        st, si = self.ost[:m].get(), self.oid[:m].get()
        o = np.lexsort((si, st))
        return st[o], si[o]
