"""Export one offline bootloader run for the README illustration."""

import json
import subprocess
from itertools import accumulate
from pathlib import Path

from cyberfly.firmware import build
from cyberfly.iss import ISS, sext
from cyberfly.llm import ask, compile_program
from cyberfly.radio import ORIGIN, Link, ask_cycle, program_frames, schedule

OUTPUT = Path(__file__).with_name("boot_trace.json")


def store_address(iss: ISS):
    """Address of a 32-bit store, if the next instruction is one."""
    instruction = iss.mem[iss.pc // 4]
    if instruction & 0x707F != 0x2023:
        return None
    offset = ((instruction >> 25) << 5) | ((instruction >> 7) & 31)
    return (iss.r[(instruction >> 15) & 31] + sext(offset, 12)) & 0xFFFFFFFF


def capture():
    boot = build("boot.c")
    actions = ask("walk a square, then rest", offline=True)
    program = compile_program(actions)
    frames = program_frames(program)
    gaps = schedule(boot, frames)
    first = ask_cycle(boot) + 120
    arrivals = list(accumulate(gaps[:-1], initial=first))
    link = Link.paced(frames, gaps, first, jitter=0)
    iss = ISS(boot, io_in=lambda: link.load(2 * iss.steps),
              on_store=lambda value: link.store(2 * iss.steps, value))
    writes = []
    request = None
    while iss.pc != ORIGIN:
        if iss.halted or 2 * iss.steps > first + sum(gaps) + 400:
            raise RuntimeError("The bootloader did not reach the uploaded program")
        address = store_address(iss)
        iss.step()
        cycle = 2 * iss.steps
        if request is None and link.sent:
            request = cycle
        if address is not None and ORIGIN <= address < ORIGIN + 4 * len(program):
            writes.append(dict(cycle=cycle, address=hex(address), word=f"0x{iss.mem[address // 4]:08x}"))
    entry = 2 * iss.steps
    assert link.sent == [ord("?")] and link.overlaps == 0
    assert [int(w["address"], 16) for w in writes] == list(range(ORIGIN, ORIGIN + 4 * len(program), 4))
    assert [int(w["word"], 16) for w in writes] == program
    iss.step()
    return dict(model="RV32I instruction-set simulator, two FPU cycles per instruction",
                source="firmware/boot.c and cyberfly.llm (fixed offline square)",
                compiler=subprocess.check_output(["clang", "--version"], text=True).splitlines()[0],
                actions=actions, boot_first_word=f"0x{boot[0]:08x}",
                request_cycle=request, frame_arrival_cycles=arrivals,
                frame_count=len(frames), header=hex(frames[0]), checksum=hex(frames[-1]),
                writes=writes, entry_address=hex(ORIGIN), entry_cycle=entry,
                cycles_from_first_frame_to_entry=entry - first,
                pc_after_entry_instruction=hex(iss.pc), t0_after_entry_instruction=hex(iss.r[5]))


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(capture(), indent=2) + "\n")
