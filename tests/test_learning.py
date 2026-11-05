import numpy as np

from cyberfly.compiler import compile_fpu
from cyberfly.firmware import build
from cyberfly.iss import ISS
from cyberfly.learning import CallCircuit, Tool, features, train_reader
from cyberfly.lif import KICK
from cyberfly.neurons import W, Kind
from cyberfly.ports import held_inputs
from cyberfly.runtime import NeuralFPU

EYE = 6
DAN_KICKS = 20


def toy_fly(seed: int = 0):
    rng = np.random.default_rng(seed)
    visual = (list(range(0, 4)), list(range(4, 8)))
    kc = list(range(8, 38))
    pre, post, count = [], [], []
    for k in kc:
        for v in rng.choice(8, 3, replace=False):
            pre.append(v), post.append(k), count.append(60)
        pre.append(k), post.append(38), count.append(1)
    tool = Tool("cpu", np.array([38]), np.array([39]), 3, 12, 30, 0.2)
    return (40, np.array(pre), np.array(post), np.array(count)), visual, kc, tool


def frames(n: int, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = np.full((n, 2, EYE), 0.6, np.float32)
    x[y == 0, :, :3] = 0.1
    x[y == 1, :, 3:] = 0.1
    return x + rng.normal(0, 0.03, x.shape).astype(np.float32), y


def train(shuffle: bool, trials: int = 120):
    fly, visual, kc, tool = toy_fly()
    x, y = frames(200, 1)
    F = features(x, np.full((2, EYE), 0.6, np.float32))
    circuit = CallCircuit(fly, visual, kc, [tool], train_reader(F[:150], y[:150], 2), eye_size=EYE, reader_init=8)
    circuit.reference["cpu"] = DAN_KICKS
    test = F[150:][y[150:] == 0]
    before = circuit.call_rates(test)["cpu"]
    rng = np.random.default_rng(3)
    for t in range(trials):
        k = int(rng.integers(0, 150))
        spikes = circuit.look(F[k], t, explore_hz=2.0)
        pays = rng.random() < 0.3 if shuffle else y[k] == 0
        dan = None
        if pays:
            dan = (circuit.sim.t + 50 * np.arange(DAN_KICKS), np.full(DAN_KICKS, 39, np.int32))
        circuit.reinforce(spikes, circuit.respond(F[k], t, {"cpu"}, extra=dan))
    return before, circuit.call_rates(test)["cpu"], circuit.call_rates(F[150:][y[150:] == 1])["cpu"]


def test_call_rate_rises_with_dopamine():
    before, after, other = train(shuffle=False)
    assert after >= before + 0.5 and other <= 0.2


def test_call_rate_flat_with_shuffled_dopamine():
    before, after, _ = train(shuffle=True)
    assert after <= before + 0.1


def test_reward_firmware():
    program = build("reward.c")
    cases = {
        1 << 3 | 1 << 4: 1 << 30 | 1 << 12 | 0x00F,
        1 << 3 | 1 << 5: 1 << 30 | 1 << 12 | 0x0F0,
        1 << 3 | 1 << 6: 1 << 30,
        1 << 10 | 1 << 9: 1 << 29 | 1 << 13,
        1 << 10 | 1 << 4: 1 << 29,
    }
    for inputs, expected in cases.items():
        iss = ISS(program, io_in=inputs).run(60)
        assert iss.io_out[:2] == [expected, 0]
    assert ISS(program, io_in=0).run(200).io_out == []


def reward_spikes(netlist, sign_bit: int) -> int:
    fpu = compile_fpu(netlist)
    fpu.load(build("reward.c"))
    entry = held_inputs(fpu, [3], 30, "call")[3]
    follower = fpu.follow({"reward": netlist.names["io_out"][12]})["reward"]
    dan = fpu.circuit.new(Kind.FOLLOW, 0, "dan")
    fpu.circuit.edge(follower, dan, W)
    neural = NeuralFPU(fpu)
    neural.record[dan] = 1
    steps = neural.cycle_steps
    sign = [(c * steps + steps // 2, fpu.inputs[sign_bit]) for c in range(70)]
    kicks = sorted(sign + [(20 * steps, entry)])
    t, i = zip(*kicks)
    _, si = neural.spikes(70, (np.array(t), np.array(i, np.int32), np.full(len(t), KICK, np.float32)))
    return int((si == dan).sum())


def test_reward_neurons(fpu_netlist):
    assert reward_spikes(fpu_netlist, 4) > 50
    assert reward_spikes(fpu_netlist, 6) == 0
