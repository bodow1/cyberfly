#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
  int32_t n;
  const int64_t *indptr;
  const int32_t *post;
  const float *w;
  double *v, *g;
  int64_t *last;
  int32_t *rfc;
  uint8_t *active;
  int32_t *act, n_act;
  int32_t delay;
  int32_t **ring, *ring_n, *ring_cap;
  int64_t t;
  double v0, vrst, vth, em, es, c, eps;
} Sim;

static void activate(Sim *s, int32_t i) {
  if (!s->active[i]) { s->active[i] = 1; s->act[s->n_act++] = i; }
}

static int refractory(const Sim *s, int64_t t, int32_t i) { return t - s->last[i] < s->rfc[i]; }

Sim *sim_new(int32_t n, const int64_t *indptr, const int32_t *post, const float *w, double v0, double vrst,
             double vth, double tmbr, double tau, double dt, int32_t delay, int32_t rfc, double eps) {
  Sim *s = calloc(1, sizeof(Sim));
  s->n = n; s->indptr = indptr; s->post = post; s->w = w;
  s->v = malloc(sizeof(double) * n); s->g = calloc(n, sizeof(double));
  s->last = malloc(sizeof(int64_t) * n); s->rfc = malloc(sizeof(int32_t) * n);
  for (int32_t i = 0; i < n; i++) { s->v[i] = v0; s->last[i] = -1000000000; s->rfc[i] = rfc; }
  s->active = calloc(n, 1); s->act = malloc(sizeof(int32_t) * n);
  s->delay = delay;
  s->ring = calloc(delay + 1, sizeof(int32_t *));
  s->ring_n = calloc(delay + 1, sizeof(int32_t));
  s->ring_cap = calloc(delay + 1, sizeof(int32_t));
  s->v0 = v0; s->vrst = vrst; s->vth = vth; s->eps = eps;
  s->em = exp(-dt / tmbr); s->es = exp(-dt / tau);
  s->c = tau / (tau - tmbr) * (s->es - s->em);
  return s;
}

int64_t sim_time(const Sim *s) { return s->t; }

void sim_voltage(const Sim *s, int32_t m, const int32_t *ids, double *out) {
  for (int32_t k = 0; k < m; k++) out[k] = s->v[ids[k]];
}

void sim_reset(Sim *s) {
  for (int32_t i = 0; i < s->n; i++) { s->v[i] = s->v0; s->g[i] = 0; s->last[i] = -1000000000; s->active[i] = 0; }
  for (int k = 0; k <= s->delay; k++) s->ring_n[k] = 0;
  s->n_act = 0;
  s->t = 0;
}

void sim_quiet(Sim *s, const uint8_t *mask) {
  for (int32_t i = 0; i < s->n; i++)
    if (mask[i]) { s->v[i] = s->v0; s->g[i] = 0; s->last[i] = -1000000000; }
  for (int k = 0; k <= s->delay; k++) {
    int32_t m = 0;
    for (int32_t j = 0; j < s->ring_n[k]; j++)
      if (!mask[s->ring[k][j]]) s->ring[k][m++] = s->ring[k][j];
    s->ring_n[k] = m;
  }
}

static void push(Sim *s, int slot, int32_t i) {
  if (s->ring_n[slot] == s->ring_cap[slot]) {
    s->ring_cap[slot] = s->ring_cap[slot] ? 2 * s->ring_cap[slot] : 1024;
    s->ring[slot] = realloc(s->ring[slot], sizeof(int32_t) * s->ring_cap[slot]);
  }
  s->ring[slot][s->ring_n[slot]++] = i;
}

int64_t sim_run(Sim *s, int64_t steps, int64_t nk, const int64_t *kt, const int32_t *ki, const float *ka,
                const uint8_t *rec, int64_t cap, int64_t *out_t, int32_t *out_i) {
  int r = s->delay + 1;
  int64_t nrec = 0, kp = 0;
  int32_t *spk = malloc(sizeof(int32_t) * (s->n > 0 ? s->n : 1));
  for (int64_t step = 0; step < steps; step++) {
    int64_t t = s->t;
    int nsp = 0;
    for (int32_t a = 0, n = s->n_act; a < n; a++) {
      int32_t i = s->act[a];
      if (refractory(s, t, i)) continue;
      double u = (s->v[i] - s->v0) * s->em + s->g[i] * s->c;
      s->g[i] *= s->es;
      s->v[i] = u + s->v0;
      if (s->v[i] > s->vth) spk[nsp++] = i;
    }
    int slot = (int)(t % r);
    for (int32_t k = 0; k < s->ring_n[slot]; k++) {
      int32_t i = s->ring[slot][k];
      for (int64_t e = s->indptr[i]; e < s->indptr[i + 1]; e++) {
        int32_t j = s->post[e];
        if (refractory(s, t, j)) continue;
        s->g[j] += s->w[e];
        activate(s, j);
      }
    }
    s->ring_n[slot] = 0;
    while (kp < nk && kt[kp] < t) kp++;
    for (; kp < nk && kt[kp] == t; kp++) {
      int32_t i = ki[kp];
      if (!refractory(s, t, i)) { s->v[i] += ka[kp]; activate(s, i); }
    }
    int out = (int)((t + s->delay) % r);
    for (int k = 0; k < nsp; k++) {
      int32_t i = spk[k];
      s->v[i] = s->vrst; s->g[i] = 0.0; s->last[i] = t;
      push(s, out, i);
      if (!rec || rec[i]) {
        if (nrec < cap) { out_t[nrec] = t; out_i[nrec] = i; }
        nrec++;
      }
    }
    if (s->eps > 0) {
      int32_t m = 0;
      for (int32_t a = 0; a < s->n_act; a++) {
        int32_t i = s->act[a];
        if (t + 1 - s->last[i] >= s->rfc[i] && fabs(s->v[i] - s->v0) < s->eps && fabs(s->g[i]) < s->eps) {
          s->v[i] = s->v0; s->g[i] = 0.0; s->active[i] = 0;
        } else {
          s->act[m++] = i;
        }
      }
      s->n_act = m;
    }
    s->t++;
  }
  free(spk);
  return nrec;
}
