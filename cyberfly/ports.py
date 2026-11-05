from .compiler import FPU
from .neurons import W, Kind

COPY_PULSES = 8


def inverted(fpu: FPU, net) -> int:
    return fpu.circuit.latches[fpu.latches[net][0]][1]


def relay(fpu: FPU, name: str) -> int:
    return fpu.circuit.new(Kind.FOLLOW, 0, name)


def hold(fpu: FPU, sources: list[int], name: str) -> int:
    c = fpu.circuit
    li = c.latch(0, name)
    for src in sources:
        s = c.new(Kind.GATE, 0, f"{name}.set")
        c.edge(src, s, W)
        c.latch_set(li, s)
    return li


def acknowledge(fpu: FPU, bit: int, latches: list[int], name: str):
    c, names = fpu.circuit, fpu.netlist.names
    k = fpu.sample_phase + 4
    k += k % 2
    chain = [c.gate(k, f"{name}.ack")]
    for v in (inverted(fpu, names["io_wr"][0]), inverted(fpu, names["io_out"][bit])):
        c.edge(v, chain[0], -W)
    for m in range(6):
        n = relay(fpu, f"{name}.ack{m + 1}")
        c.edge(chain[-1], n, W)
        chain.append(n)
    for m in (0, 2, 4, 6):
        r = c.new(Kind.GATE, 0, f"{name}.reset{m}")
        c.edge(chain[m], r, W)
        for li in latches:
            c.latch_reset(li, r)


def copy_into_port(fpu: FPU, latches: dict[int, int], name: str):
    c, port = fpu.circuit, fpu.netlist.port_bits
    k = fpu.clear_phases[-1] + 3
    k += k % 2
    assert k + 2 * (COPY_PULSES - 1) < fpu.M + min(fpu.input_samples) - 2
    for bit, li in latches.items():
        g = c.new(Kind.GATE, k, f"{name}.copy{bit}")
        for j in range(k, k + 2 * COPY_PULSES, 2):
            c.edge(c.clock(j - 1), g, W)
        c.edge(c.latches[li][1], g, -W)
        for target in fpu.latches[port[bit]]:
            c.latch_set(target, g)


def held_inputs(fpu: FPU, bits: list[int], ack_bit: int, name: str = "held") -> dict[int, int]:
    entries, latches = {}, {}
    for bit in bits:
        entry = relay(fpu, f"{name}{bit}")
        echo = relay(fpu, f"{name}{bit}.echo")
        fpu.circuit.edge(entry, echo, W)
        entries[bit] = entry
        latches[bit] = hold(fpu, [entry, echo], f"{name}{bit}")
    acknowledge(fpu, ack_bit, list(latches.values()), name)
    copy_into_port(fpu, latches, name)
    return entries
