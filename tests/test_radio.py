import numpy as np
import pytest

from cyberfly.compiler import compile_fpu
from cyberfly.firmware import build
from cyberfly.iss import ISS
from cyberfly.lif import KICK
from cyberfly.llm import ask, compile_program, finished
from cyberfly.logic import GateMachine
from cyberfly.radio import ORIGIN, Link, add_radio, ask_cycle, decode, frame_spikes, program_frames, schedule
from cyberfly.runtime import NeuralFPU, period


@pytest.fixture(scope="module")
def upload():
    boot = build("boot.c")
    program = compile_program(ask("walk a square, then rest", offline=True))
    frames = program_frames(program)
    gaps = schedule(boot, frames)
    start = ask_cycle(boot) + 120
    return boot, program, frames, gaps, start, start + sum(gaps) + 400


def test_frames_round_trip():
    p = period()
    for word in (0, 1, 0x3FFF, 0x2A55, ord("?")):
        assert decode(frame_spikes(word, 100, p), p) == [(100, word)]


def test_bootloader_iss(upload):
    boot, program, frames, gaps, start, end = upload
    link = Link.paced(frames, gaps, start)
    iss = ISS(boot, io_in=lambda: link.load(2 * iss.steps), on_store=lambda v: link.store(2 * iss.steps, v))
    jumped = None
    while 2 * iss.steps < end:
        iss.step()
        if jumped is None and iss.pc >= ORIGIN:
            jumped = 2 * iss.steps
    assert link.sent == [ord("?")] and link.overlaps == 0
    assert jumped is not None and jumped > start
    assert iss.mem[ORIGIN // 4:ORIGIN // 4 + len(program)] == program
    assert finished(iss.io_out[-1])


def test_bootloader_gates(fpu_netlist, upload):
    boot, program, frames, gaps, start, end = upload
    link = Link.paced(frames, gaps, start, seed=1)
    gate = GateMachine(fpu_netlist, boot)
    io_wr = fpu_netlist.names["io_wr"][0]
    jumped = None
    for cycle in range(end):
        gate.io_in = link.load(cycle)
        gate.step()
        if gate.latch(io_wr):
            link.store(cycle, gate.latched("io_out"))
        if jumped is None and gate.latched("pc") >= ORIGIN:
            jumped = cycle
    assert link.sent == [ord("?")] and link.overlaps == 0
    assert jumped is not None and jumped > start
    assert gate.mem[ORIGIN // 4:ORIGIN // 4 + len(program)] == program
    assert finished(gate.latched("io_out"))


def test_radio_neurons_echo(fpu_netlist):
    fpu = compile_fpu(fpu_netlist)
    fpu.load(build("echo.c"))
    radio = add_radio(fpu)
    neural = NeuralFPU(fpu)
    neural.record[[radio.tx, radio.rx]] = 1
    p = period()
    word = 0x1ABC
    t = frame_spikes(word, neural.sample_time(8), p)
    kicks = (t - 1, np.full(len(t), radio.rx, np.int32), np.full(len(t), KICK, np.float32))
    st, si = neural.spikes(60, kicks)
    assert decode(st[si == radio.rx], p) == [(int(t[0]), word)]
    assert [w for _, w in decode(st[si == radio.tx], p)] == [word + 1]
