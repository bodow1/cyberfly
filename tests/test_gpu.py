from cyberfly.iss import ISS
from cyberfly.logic import GateMachine
from cyberfly.raycaster import Raycaster, render


def test_shooter_frames_match_reference(gpu_netlist, shooter):
    gate = GateMachine(gpu_netlist, shooter)
    names = gpu_netlist.names
    io_wr, dist = names["io_wr"][0], names["dist"]
    frames = []
    for _ in range(2500):
        gate.step()
        if gate.latch(io_wr):
            cam = gate.latched("cam")
            drawn = [sum(gate.latch(dist[c * 5 + j]) << j for j in range(5)) for c in range(32)]
            assert drawn == render(cam & 0xFFF, (cam >> 12) & 0xFFF, (cam >> 24) & 63)
            frames.append(gate.latched("io_out"))
    iss = ISS(shooter, gpu=Raycaster()).run(10_000)
    assert len(frames) >= 8
    assert frames == iss.io_out[:len(frames)]


def test_raycaster_sees_walls():
    d = render(0x780, 0x880, 0)
    assert len(d) == 32 and min(d) >= 1 and max(d) <= 31
    assert d != render(0x780, 0x880, 16)
