from .compiler import FPU
from .neurons import W, Kind

COPY_PULSES = 8


def inverted(fpu: FPU, net) -> int:
    return fpu.circuit.latches[fpu.latches[net][0]][1]


def relay(fpu: FPU, name: str) -> int:
    return fpu.circuit.new(Kind.FOLLOW, 0, name)


def entry(fpu: FPU, name: str) -> tuple[int, int]:
    e, echo = relay(fpu, name), relay(fpu, f"{name}.echo")
    fpu.circuit.edge(e, echo, W)
    return e, echo


def hold(fpu: FPU, sources: tuple[int, int], name: str) -> int:
    c = fpu.circuit
    li = c.latch(0, name)
    for src in sources:
        s = c.new(Kind.GATE, 0, f"{name}.set")
        c.edge(src, s, W)
        c.latch_set(li, s)
    return li


def clear(fpu: FPU, sources: tuple[int, int], latches: list[int], name: str):
    c = fpu.circuit
    for src in sources:
        r = c.new(Kind.GATE, 0, f"{name}.clear")
        c.edge(src, r, W)
        for li in latches:
            c.latch_reset(li, r)


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


def held_call(fpu: FPU, bit: int, name: str) -> tuple[int, int]:
    pair = entry(fpu, f"{name}{bit}")
    li = hold(fpu, pair, f"{name}{bit}")
    copy_into_port(fpu, {bit: li}, name)
    return pair[0], li


def held_choice(fpu: FPU, bits: list[int], name: str) -> tuple[dict[int, int], dict[int, int]]:
    pairs = {bit: entry(fpu, f"{name}{bit}") for bit in bits}
    latches = {bit: hold(fpu, pair, f"{name}{bit}") for bit, pair in pairs.items()}
    for bit, pair in pairs.items():
        clear(fpu, pair, [li for b, li in latches.items() if b != bit], f"{name}{bit}")
    copy_into_port(fpu, latches, name)
    return {bit: pair[0] for bit, pair in pairs.items()}, latches
