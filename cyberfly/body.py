from dataclasses import dataclass

import numpy as np

from .brain import LC10C
from .connectome import Connectome
from .lif import KICK, PARAMS
from .neurons import W

OMMATIDIA = 721
MAX_RATE = 150.0
STEPS_PER_EXCHANGE = 100


@dataclass
class Eyes:
    first: int
    pre: np.ndarray
    post: np.ndarray

    @classmethod
    def wire(cls, fly: Connectome, first: int, seed: int = 0) -> "Eyes":
        rng = np.random.default_rng(seed)
        lc10 = {0: fly.cells(LC10C, "left"), 1: fly.cells(LC10C, "right")}
        lc4 = fly.cells(["LC4"])
        pre, post = [], []
        for eye in (0, 1):
            for k in range(OMMATIDIA):
                p = first + eye * OMMATIDIA + k
                for c in rng.choice(lc10[eye], 2, replace=False):
                    pre.append(p)
                    post.append(int(c))
                if rng.random() < 0.3:
                    pre.append(p)
                    post.append(int(rng.choice(lc4)))
        return cls(first, np.array(pre), np.array(post))

    @property
    def edges(self):
        return self.pre, self.post, np.full(len(self.pre), W)

    def kicks(self, darkness: np.ndarray, t0: int, rng: np.random.Generator):
        p = darkness.reshape(-1) * MAX_RATE * PARAMS["dt"] * 1e-3
        steps, pix = np.nonzero(rng.random((STEPS_PER_EXCHANGE, p.size)) < p)
        return (steps.astype(np.int64) + t0, (self.first + pix).astype(np.int32), np.full(len(steps), KICK, np.float32))


def darkness(vision: np.ndarray, calibration: np.ndarray, sky: np.ndarray) -> np.ndarray:
    bright = np.asarray(vision, float).mean(axis=2)
    return sky * np.clip(((calibration - bright) / calibration - 0.15) / 0.45, 0, 1)


@dataclass
class Motor:
    forward: float = 0.6
    walk_turn: float = 0.1
    pivot: float = 0.45
    stop_hz: float = 25.0
    rates: dict | None = None

    def update(self, counts: dict, sizes: dict, seconds: float):
        self.rates = self.rates or {k: 0.0 for k in counts}
        for k, n in counts.items():
            self.rates[k] = 0.8 * self.rates[k] + 0.2 * n / sizes[k] / seconds

    def action(self) -> np.ndarray:
        r = self.rates
        turn = float(np.clip((r["DNa02_L"] - r["DNa02_R"]) / 60.0, -1, 1))
        if r["GF"] > self.stop_hz:
            return np.array([-self.pivot * turn, self.pivot * turn]) if abs(turn) > 0.15 else np.zeros(2)
        return np.array([self.forward * (1 - self.walk_turn * turn), self.forward * (1 + self.walk_turn * turn)])


def world(camera: str = "camera_top_zoomout"):
    from flygym import Camera, Fly
    from flygym.arena import FlatTerrain
    from flygym.examples.locomotion import HybridTurningController
    from flygym.preprogrammed import all_leg_dofs

    arena = FlatTerrain(size=(3000, 3000), ground_alpha=1.0)
    for tex in arena.root_element.asset.find_all("texture"):
        if getattr(tex, "builtin", None) in ("checker", "gradient") or tex.name in ("grid", "groundplane"):
            tex.rgb1 = tex.rgb2 = (0.82, 0.80, 0.76)
    legs = [f"{s}{p}" for s in "LR" for p in "FMH"]
    sensors = [f"{leg}{seg}" for leg in legs for seg in ["Tibia"] + [f"Tarsus{i}" for i in range(1, 6)]]
    fly = Fly(init_pose="stretch", actuated_joints=all_leg_dofs, control="position",
              contact_sensor_placements=sensors, enable_adhesion=True, draw_adhesion=False,
              enable_vision=True, render_raw_vision=False, vision_refresh_rate=100)
    cam = Camera(attachment_point=fly.model.worldbody, camera_name=camera, targeted_fly_names=fly.name,
                 play_speed=0.5, window_size=(960, 720))
    return HybridTurningController(fly=fly, cameras=[cam], arena=arena, timestep=1e-4), cam


def calibrate(sim, settle: int = 300, samples: int = 5):
    obs = None
    for _ in range(settle):
        obs, *_ = sim.step(np.zeros(2))
    frames = []
    for _ in range(samples):
        for _ in range(20):
            obs, *_ = sim.step(np.zeros(2))
        frames.append(np.asarray(obs["vision"], float).mean(axis=2))
    cal = np.mean(frames, axis=0)
    return obs, cal, cal > 0.9 * cal.max(axis=1, keepdims=True)
