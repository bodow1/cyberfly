import json
import shutil
import subprocess
import tempfile
from functools import cached_property
from pathlib import Path

HW = Path(__file__).resolve().parent.parent / "hw"

DESIGNS = {
    "fly_fpu": (["neural.v", "fpu.v", "top.v"], ["NEURAL_RAM"]),
    "fly_fpu_gpu": (["neural.v", "fpu.v", "gpu.v", "top.v"], ["NEURAL_RAM", "GPU_PORT"]),
    "fpu_ext": (["neural.v", "fpu.v", "top.v"], ["EXT_MEM"]),
}


def synthesize(files: list[Path], top: str, defines: list[str] = ()) -> dict:
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            shutil.copy(f, td)
        names = " ".join(Path(f).name for f in files)
        flags = " ".join(f"-D{d}" for d in defines)
        script = (
            f"read_verilog {flags} -sv {names}; hierarchy -top {top}; synth -flatten -top {top}; "
            "async2sync; dffunmap; abc -g NOR; opt_clean; setundef -zero; opt_clean; write_json out.json"
        )
        r = subprocess.run(["yowasp-yosys", "-q", "-p", script], capture_output=True, text=True, cwd=td)
        if r.returncode:
            raise RuntimeError(r.stdout[-3000:] + r.stderr[-3000:])
        return json.loads((Path(td) / "out.json").read_text())


class Netlist:
    def __init__(self, yosys_json: dict, top: str):
        m = yosys_json["modules"][top]
        self.gates: list[tuple] = []
        self.dffs: list[tuple] = []
        self.rams: list[dict] = []
        self.ports: list[list] = []
        init = {}
        for nd in m["netnames"].values():
            if "init" in nd.get("attributes", {}):
                v = nd["attributes"]["init"]
                bits = v if isinstance(v, str) else format(int(v), "b").zfill(len(nd["bits"]))
                for b, ch in zip(nd["bits"], reversed(bits)):
                    if isinstance(b, int):
                        init[b] = int(ch == "1")
        for cell in m["cells"].values():
            t, p = cell["type"], cell["connections"]
            if t == "$_NOR_":
                self.gates.append((p["Y"][0], [p["A"][0], p["B"][0]]))
            elif t == "$_NOT_":
                self.gates.append((p["Y"][0], [p["A"][0]]))
            elif t in ("$_DFF_P_", "$_DFF_N_"):
                self.dffs.append((p["D"][0], p["Q"][0], init.get(p["Q"][0], 0)))
            elif t.endswith("neural_port"):
                self.ports.append(p["v"])
            elif "neural_ram" in t:
                self.rams.append(dict(addr=p["addr"], we=p["we"][0], wdata=p["wdata"], rdata=p["rdata"]))
            elif t != "$scopeinfo":
                raise ValueError(f"unsupported cell {t}")
        self.names = {nm: nd["bits"] for nm, nd in m["netnames"].items()}

    @classmethod
    def build(cls, design: str) -> "Netlist":
        files, defines = DESIGNS[design]
        return cls(synthesize([HW / f for f in files], design, defines), design)

    @cached_property
    def topo(self) -> list[tuple]:
        drv = dict(self.gates)
        seen, order = set(), []
        for root in drv:
            stack = [(root, False)]
            while stack:
                x, done = stack.pop()
                if x in seen or x not in drv:
                    continue
                if done:
                    seen.add(x)
                    order.append((x, drv[x]))
                    continue
                stack.append((x, True))
                stack.extend((i, False) for i in drv[x] if i not in seen and i in drv)
        return order

    def init_state(self) -> dict:
        return {q: v for _, q, v in self.dffs}

    @property
    def port_bits(self) -> list:
        return [b for port in self.ports for b in port if not isinstance(b, str)]
