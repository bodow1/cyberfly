import os
import subprocess
from pathlib import Path

import pytest

from cyberfly.compiler import compile_fpu
from cyberfly.firmware import words
from cyberfly.hdl import Netlist
from cyberfly.iss import ISS
from cyberfly.logic import GateMachine
from cyberfly.runtime import NeuralFPU

FIB = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 1002, 1003, 1005, 1007, 1011, 1013, 1017, 1019, 1023, 1029,
       1031, 1037]


def test_iss_runs_c(fib):
    assert ISS(fib).run().io_out == FIB


def test_gates_match_iss(fpu_netlist, fib):
    gate, iss = GateMachine(fpu_netlist, fib), ISS(fib)
    names = fpu_netlist.names
    regs = [f"core.regs[{i}]" for i in range(1, 32)]
    outputs = []
    while not iss.halted:
        gate.step()
        gate.step()
        iss.step()
        if gate.latch(names["io_wr"][0]):
            outputs.append(gate.latched("io_out"))
        assert gate.latched("pc") == iss.pc
        assert [gate.latched(r) if r in names else 0 for r in regs] == iss.r[1:]
    assert outputs == FIB


def test_compiled_size(fpu_netlist):
    fpu = compile_fpu(fpu_netlist)
    stats = fpu.circuit.stats()
    assert fpu.M == 132
    assert stats["latch_a"] == stats["latch_b"] == len(fpu_netlist.dffs) + 32 * 256 + 32


def test_neural_fpu_matches_gates(fpu_netlist, fib):
    fpu = compile_fpu(fpu_netlist)
    fpu.load(fib)
    neural, gate = NeuralFPU(fpu), GateMachine(fpu_netlist, fib)
    io_wr, io_out = fpu_netlist.names["io_wr"][0], fpu_netlist.names["io_out"]
    mismatches, outputs = 0, []
    for start in range(0, 50, 10):
        spikes = neural.run(10)
        for cycle in range(start, start + 10):
            state = neural.states(spikes, neural.watch, cycle)
            mismatches += sum(state[q] != gate.latch(q) for q in neural.watch)
            if cycle % 25 == 0:
                ram = neural.ram(spikes, cycle)
                mismatches += sum(v != (gate.mem[w] >> b) & 1 for (w, b), v in ram.items())
            if state[io_wr]:
                outputs.append(sum(state[b] << i for i, b in enumerate(io_out)))
            gate.step()
    assert mismatches == 0
    assert outputs == FIB[:len(outputs)] and len(outputs) >= 4


RISCV_TESTS = os.environ.get("RISCV_TESTS")
SKIP = {"fence_i", "ma_data"}


def rv32ui_images(root: Path, out: Path) -> dict[str, list[int]]:
    env = Path(__file__).resolve().parent.parent / "firmware" / "riscv-tests"
    images = {}
    for src in sorted((root / "isa" / "rv32ui").glob("*.S")):
        if src.stem in SKIP:
            continue
        elf, binary = out / f"{src.stem}.elf", out / f"{src.stem}.bin"
        subprocess.run(["clang", "--target=riscv32", "-march=rv32i", "-mabi=ilp32", "-nostdlib", f"-I{env}",
                        f"-I{root / 'isa' / 'macros' / 'scalar'}", "-Wl,-Ttext=0", "-fuse-ld=lld", "-Wl,--no-relax",
                        str(src), "-o", str(elf)], check=True)
        subprocess.run(["llvm-objcopy", "-O", "binary", str(elf), str(binary)], check=True)
        images[src.stem] = words(binary.read_bytes())
    return images


@pytest.mark.skipif(not RISCV_TESTS, reason="set RISCV_TESTS to a riscv-tests checkout")
def test_rv32ui(tmp_path):
    nl = Netlist.build("fpu_ext")
    failed = []
    for name, image in rv32ui_images(Path(RISCV_TESTS), tmp_path).items():
        gate = GateMachine(nl, [])
        ev, mem, result = gate.ev, image + [0] * 16384, None
        for cycle in range(40_000):
            ev.force_bus("ext_rdata", 0)
            ev.eval(gate.qv)
            ev.force_bus("ext_rdata", mem[ev.bus("ext_widx")])
            nq = ev.eval(gate.qv)
            if ev.bus("ext_we"):
                mem[ev.bus("ext_widx")] = ev.bus("ext_wdata")
            if ev.bus("io_wr") and result is None and cycle % 2 == 0:
                result = ev.bus("io_out")
            if ev.bus("halted"):
                break
            gate.qv = nq
        if result != 1:
            failed.append(name)
    assert not failed
