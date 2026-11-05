import json
import os
import urllib.request

from .firmware import assemble
from .radio import ORIGIN

ACTIONS = {"straight": 0x000, "left": 0x00F, "right": 0x0F0, "stop": 0xF00}
SIGNATURE = 0xC0 << 16
REGISTERS = {"straight": "s1", "left": "s2", "right": "s3", "stop": "s4"}
SECONDS_PER_INSTRUCTION = 2 * 132 * 49 * 1e-4
TURN_RATE = {"left": 22.0, "right": 20.0}
UNIT_SECONDS, UNIT_DEGREES = 5.0, 30.0
MAX_ACTIONS = 12

SYSTEM = (
    "You write programs for a walking fruit fly. Reply with JSON only: "
    '{"program": [{"act": "straight" | "left" | "right" | "stop", "units": 1-6}, ...]}. '
    "One unit of straight is about 5 seconds of walking. One unit of left or right is about 30 degrees of turning "
    "while walking, so 3 units is a quarter turn. stop makes the fly stand still and ends the program. "
    f"Use at most {MAX_ACTIONS} actions."
)

OFFLINE = [
    {"act": "straight", "units": 2}, {"act": "left", "units": 3},
    {"act": "straight", "units": 2}, {"act": "left", "units": 3},
    {"act": "straight", "units": 2}, {"act": "left", "units": 3},
    {"act": "straight", "units": 2}, {"act": "stop", "units": 1},
]


def ask(request: str, offline: bool | None = None) -> list[dict]:
    base = os.environ.get("CYBERFLY_LLM_BASE_URL")
    if offline or (offline is None and not base):
        return [dict(a) for a in OFFLINE]
    body = {
        "model": os.environ.get("CYBERFLY_LLM_MODEL", "gpt-4o-mini"),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": request}],
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('CYBERFLY_LLM_API_KEY', '')}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        content = json.load(r)["choices"][0]["message"]["content"]
    return validate(json.loads(content)["program"])


def validate(program: list[dict]) -> list[dict]:
    out = []
    for a in program[:MAX_ACTIONS]:
        act, units = str(a["act"]), max(1, min(6, int(a["units"])))
        if act not in ACTIONS:
            raise ValueError(f"unknown action {act!r}")
        out.append({"act": act, "units": units})
        if act == "stop":
            break
    return out


def to_asm(program: list[dict]) -> str:
    lines = ["  .text", "  .globl entry", "entry:", "  lui t0, 0x80000"]
    lines += [f"  li {r}, {ACTIONS[a] | SIGNATURE:#x}" for a, r in REGISTERS.items()]
    for a in program:
        act, units = a["act"], a["units"]
        lines.append(f"  sw {REGISTERS[act]}, 0(t0)")
        if act == "stop":
            break
        if act == "straight":
            n = max(1, round((units * UNIT_SECONDS / SECONDS_PER_INSTRUCTION - 2) / 2))
            lines += [f"  li t1, {n}", "1: addi t1, t1, -1", "  bnez t1, 1b"]
        else:
            n = max(0, round(units * UNIT_DEGREES / TURN_RATE[act] / SECONDS_PER_INSTRUCTION) - 1)
            lines += ["  nop"] * n
    lines += ["  sw s4, 0(t0)", "2: j 2b"]
    return "\n".join(lines) + "\n"


def compile_program(program: list[dict]) -> list[int]:
    return assemble(to_asm(program), ORIGIN)


def finished(io_out: int) -> bool:
    return io_out == ACTIONS["stop"] | SIGNATURE
