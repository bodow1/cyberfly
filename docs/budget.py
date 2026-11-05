from collections import Counter

from cyberfly.compiler import compile_fpu
from cyberfly.hdl import Netlist
from cyberfly.neurons import Kind

RAM = ("qm", "nrg", "set", "nnwl", "ndec", "clr", "sel", "nsel", "ncol", "nna")


def part(name: str, kind: int, ports: set) -> str:
    if kind in (Kind.RING, Kind.CLOCK, Kind.OSC):
        return "clock ring, clock copies and oscillators"
    if kind == Kind.CAPTURE or name == "inclr" or name.split(".")[0] in ports:
        return "input port"
    if name.startswith(("qm", "nrg", "set")):
        return "RAM bits"
    if name.startswith(RAM):
        return "RAM address decoding and write select"
    if kind in (Kind.LATCH_A, Kind.LATCH_B) or name[:1] in "SR":
        return "registers and other flip-flops"
    if "~" in name:
        return "delay relays and fan-out copies"
    return "CPU logic gates"


def main():
    nl = Netlist.build("fly_fpu")
    c = compile_fpu(nl).circuit
    ports = {f"q{b}" for b in nl.port_bits}
    counts = Counter(part(n, k, ports) for n, k in zip(c.name, c.kind))
    for name, n in counts.most_common():
        print(f"| {name} | {n:,} |")
    print(f"| total | {sum(counts.values()):,} |")


if __name__ == "__main__":
    main()
