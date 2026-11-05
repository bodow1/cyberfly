from collections import defaultdict
from dataclasses import dataclass, field

from .hdl import Netlist
from .neurons import FANOUT, W, Circuit, Kind

SETTLE = 16
CLEAR_GAP = 16
CLEAR_PULSES = 8


@dataclass
class FPU:
    circuit: Circuit
    M: int
    netlist: Netlist
    sample_phase: int
    latches: dict
    memory: dict
    inputs: dict
    input_samples: list
    clear_phases: list | None
    followers: dict = field(default_factory=dict)

    def latch_neuron(self, net) -> int:
        return self.circuit.latches[self.latches[net][0]][0]

    def follow(self, bits: dict) -> dict:
        c = self.circuit
        out = {}
        for name, net in bits.items():
            _, b = c.latches[self.latches[net][0]]
            f = c.new(Kind.FOLLOW, 0, f"F{name}")
            c.edge(c.oscillator(1), f, W)
            c.edge(b, f, -W)
            out[name] = f
        self.followers.update(out)
        return out

    def load(self, words: list[int]):
        c = self.circuit
        ones = {li for (w, b), lis in self.memory.items() if w < len(words) and (words[w] >> b) & 1 for li in lis}
        mem = {li for lis in self.memory.values() for li in lis}
        owner = {n: li for li, pair in enumerate(c.latches) for n in pair}
        c.boot = [(p, n) for p, n in c.boot if owner.get(n) not in mem]
        for li in mem:
            a, b = c.latches[li]
            c.boot.append((0, a) if li in ones else (1, b))


def _topological(gates: list) -> list:
    drv = dict(gates)
    seen, order = set(), []
    for root in drv:
        stack = [(root, False)]
        while stack:
            x, done = stack.pop()
            if x in seen or x not in drv:
                continue
            if done:
                seen.add(x)
                order.append((x, drv[x]))
                continue
            stack.append((x, True))
            for i in drv[x]:
                ii = _base(i)
                if ii not in seen and ii in drv:
                    stack.append((ii, False))
    return order


def _base(net):
    return net[1:] if isinstance(net, str) and net.startswith("~") else net


def compile_fpu(nl: Netlist, settle: int = SETTLE) -> FPU:
    qset = {q for _, q, _ in nl.dffs}
    extra, mem_bits, rams = [], [], []
    for mi, ram in enumerate(nl.rams):
        aw = len(ram["addr"])
        a = ram["addr"]
        for j in range(aw):
            extra.append((f"na{mi}_{j}", [a[j]]))
        for w in range(1 << aw):
            lits = [a[j] if not (w >> j) & 1 else f"na{mi}_{j}" for j in range(aw)]
            extra.append((f"dec{mi}_{w}", lits))
            extra.append((f"nwl{mi}_{w}", [f"dec{mi}_{w}"]))
            for b in range(32):
                m = f"m{mi}_{w}_{b}"
                mem_bits.append(m)
                extra.append((f"rg{mi}_{w}_{b}", [f"nwl{mi}_{w}", "~" + m]))
        for b in range(32):
            extra.append((f"col{mi}_{b}", [f"rg{mi}_{w}_{b}" for w in range(1 << aw)]))
            extra.append((ram["rdata"][b], [f"col{mi}_{b}"]))
        rams.append(dict(mi=mi, words=1 << aw, addr=a, we=ram["we"], wdata=ram["wdata"]))
    qset |= set(mem_bits)
    in_bits = nl.port_bits
    qset |= set(in_bits)

    topo = _topological(list(nl.gates) + extra)
    dffs = []
    for d, q, init in nl.dffs:
        if d == q:
            dffs.append((None, q, init))
            continue
        if d in qset:
            a, b = f"cpa{q}", f"cpb{q}"
            topo += [(a, [d]), (b, [a])]
            d = b
        dffs.append((d, q, init))

    const = {"0": 0, "1": 1}
    for out, ins in topo:
        if any(const.get(i) == 1 for i in ins):
            const[out] = 0
        elif all(i in const for i in ins):
            const[out] = 1

    ph = {}
    for out, ins in topo:
        if out in const:
            continue
        p = 2
        for i in ins:
            if i in const or _base(i) in qset:
                continue
            p = max(p, ph[i] + 1)
        ph[out] = p

    dnets = {d for d, _, _ in dffs if d is not None and d not in const}
    for r in rams:
        dnets |= {n for n in list(r["addr"]) + [r["we"]] + list(r["wdata"]) if n in ph}
    end = max([ph[d] for d in dnets] + [2])
    consumers = defaultdict(list)
    for out, ins in topo:
        if out in const:
            continue
        for i in ins:
            if i in ph:
                consumers[i].append(out)
    for out, _ in reversed(topo):
        if out in const:
            continue
        limit = min([ph[c] - 1 for c in consumers[out]] + ([end] if out in dnets or not consumers[out] else []))
        if limit > ph[out]:
            ph[out] = limit

    def tap(net, k, positive):
        p = ph[net]
        for lag in (1, 2):
            j = k - lag - p
            if j >= 0 and (j % 2 == 0) == positive:
                return ("tap", net, j), lag
        raise AssertionError((net, k, p, positive))

    def literal(net, k, positive=True):
        if isinstance(net, str) and net.startswith("~"):
            net, positive = net[1:], not positive
        if net in qset:
            return ("L", net, positive), None
        return tap(net, k, positive)

    spec = {}
    for out, ins in topo:
        if out in const:
            continue
        k = ph[out]
        spec[out] = (k, [literal(i, k) for i in ins if i not in const])

    samples = defaultdict(list)
    for out, (k, lits) in spec.items():
        for s, _ in lits:
            if s[0] == "L":
                samples[s[1]].append(k)
    writes = {}
    for _ in range(64):
        new = {}
        for d, q, _ in dffs:
            if d is None:
                new[q] = ("hold",)
                continue
            if d in const:
                new[q] = ("const", const[d])
                continue
            last = max(samples[q]) if samples[q] else 2
            pd = ph[d]
            e = max(last + 1, pd + 2)
            e += e % 2
            o = max(last + 1, pd + 1)
            o += 1 - o % 2
            new[q] = ("sr", d, e, o)
        if new == writes:
            break
        writes = new
    else:
        raise RuntimeError("write schedule did not converge")

    for r in rams:
        mi = r["mi"]
        msamp = [k for q, v in samples.items() if isinstance(q, str) and q.startswith(f"m{mi}_") for k in v]
        nets = list(r["addr"]) + [r["we"]] + list(r["wdata"])
        ready = max([ph[n] for n in nets if n in ph] + [2])
        o = max(max(msamp) + 1, ready + 2)
        o += 1 - o % 2
        e = max(o + CLEAR_GAP, ready + 4)
        e += e % 2
        r["o"], r["e"] = o, e

    clear = None
    if in_bits:
        isamp = [k for q in in_bits for k in samples.get(q, [])] or [2]
        c0 = max(isamp) + 1
        c0 += 1 - c0 % 2
        clear = list(range(c0, c0 + 2 * CLEAR_PULSES, 2))
    first = 1
    done = max(([clear[-1] + 2] if clear else [])
               + [w[2] + 2 for w in writes.values() if w[0] == "sr"]
               + [w[3] + 2 for w in writes.values() if w[0] == "sr"]
               + list(ph.values()) + [first] + [r["e"] + 2 for r in rams])
    M = done - first + settle
    M += M % 2
    c = Circuit(M)

    copies = defaultdict(list)
    lcopies = defaultdict(list)
    init_of = {q: v for _, q, v in nl.dffs}
    for m in mem_bits:
        init_of[m] = 0
    for q in in_bits:
        init_of[q] = 0

    def make_tap(key):
        _, net, j = key
        k = ph[net] + j
        g = c.gate(k, f"n{net}~{j}" if j else f"n{net}")
        if j == 0:
            for s, lag in spec[net][1]:
                connect(s, g, k, lag)
        else:
            connect(("tap", net, j - 1), g, k, 1)
        return g

    def driver(key):
        for n in copies[key]:
            if c.fanout[n] < FANOUT:
                return n
        n = make_tap(key)
        copies[key].append(n)
        return n

    def latch_neuron(q, parity):
        for li in lcopies[q]:
            n = c.latches[li][parity]
            if c.fanout[n] < FANOUT:
                return n
        li = c.latch(init_of[q], name=f"q{q}")
        lcopies[q].append(li)
        return c.latches[li][parity]

    def connect(s, g, k, lag):
        if s[0] == "L":
            net, positive = s[1], s[2]
            for lg in (1, 2):
                parity = (k - lg) % 2
                if (parity == 0) == positive:
                    c.edge(latch_neuron(net, parity), g, -W)
                    return
        n = driver(s)
        assert c.is_at(n, k - lag), (s, k, lag)
        c.edge(n, g, -W)

    def make_sr(q):
        _, d, e, o = writes[q]
        s = c.gate(e, f"S{q}")
        r = c.gate(o, f"R{q}")
        s1, l1 = literal(d, e, positive=False)
        connect(s1, s, e, l1)
        s2, l2 = literal(d, o, positive=True)
        connect(s2, r, o, l2)
        return s, r

    for out in spec:
        driver(("tap", out, 0))
    for _, q, _ in nl.dffs:
        if not lcopies[q]:
            latch_neuron(q, 0)
    srs = defaultdict(list)
    wired = defaultdict(set)
    changed = True
    while changed:
        changed = False
        for q in [q for q in writes if writes[q][0] == "sr"]:
            for li in list(lcopies[q]):
                if li in wired[q]:
                    continue
                pair = next((p for p in srs[q] if c.fanout[p[0]] < FANOUT and c.fanout[p[1]] < FANOUT), None)
                if pair is None:
                    pair = make_sr(q)
                    srs[q].append(pair)
                c.latch_set(li, pair[0])
                c.latch_reset(li, pair[1])
                wired[q].add(li)
                changed = True

    def word_match(g, k, r, w):
        if r["we"] not in const:
            s, lg = literal(r["we"], k, positive=False)
            connect(s, g, k, lg)
        for j, a in enumerate(r["addr"]):
            if a in const:
                continue
            s, lg = literal(a, k, positive=not (w >> j) & 1)
            connect(s, g, k, lg)

    memory = {}
    for r in rams:
        mi, o, e = r["mi"], r["o"], r["e"]
        for w in range(r["words"]):
            clr = c.gate(o, f"clr{mi}_{w}")
            word_match(clr, o, r, w)
            sel = c.gate(e - 2, f"sel{mi}_{w}")
            word_match(sel, e - 2, r, w)
            nsel = c.gate(e - 1, f"nsel{mi}_{w}")
            c.edge(sel, nsel, -W)
            for b in range(32):
                m = f"m{mi}_{w}_{b}"
                if not lcopies[m]:
                    latch_neuron(m, 0)
                wd = r["wdata"][b]
                s = None
                if not (wd in const and const[wd] == 0):
                    s = c.gate(e, f"set{mi}_{w}_{b}")
                    c.edge(nsel, s, -W)
                    if wd not in const:
                        sl, lg = literal(wd, e, positive=False)
                        connect(sl, s, e, lg)
                for li in lcopies[m]:
                    c.latch_reset(li, clr)
                    if s is not None:
                        c.latch_set(li, s)
                memory[(w, b)] = lcopies[m]

    inputs = {}
    if in_bits:
        rclr = c.new(Kind.GATE, clear[0], "inclr")
        for k in clear:
            c.edge(c.clock(k - 1), rclr, W)
        for i, q in enumerate(in_bits):
            if not lcopies[q]:
                latch_neuron(q, 0)
            x = c.new(Kind.CAPTURE, 0, f"cap{i}")
            for li in lcopies[q]:
                c.latch_reset(li, rclr)
                c.latch_set(li, x)
            inputs[i] = x
    in_set = set(in_bits)
    input_samples = sorted({k - lg for _, (k, lits) in spec.items() for (s, lg0) in lits
                            if s[0] == "L" and s[1] in in_set for lg in ((1, 2) if lg0 is None else (lg0,))})
    return FPU(circuit=c, M=M, netlist=nl, sample_phase=first + first % 2, latches=dict(lcopies),
               memory=memory, inputs=inputs, input_samples=input_samples, clear_phases=clear)
