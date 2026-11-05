import struct
import subprocess
import tempfile
from pathlib import Path

FIRMWARE = Path(__file__).resolve().parent.parent / "firmware"
RAM_BYTES = 1024


def words(data: bytes) -> list[int]:
    data += b"\0" * (-len(data) % 4)
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def build(source: str | Path, ram_bytes: int = RAM_BYTES) -> list[int]:
    source = Path(source)
    if not source.is_absolute() and not source.exists():
        source = FIRMWARE / source
    with tempfile.TemporaryDirectory() as td:
        elf, binary = Path(td) / "a.elf", Path(td) / "a.bin"
        subprocess.run([
            "clang", "--target=riscv32", "-march=rv32i", "-mabi=ilp32", "-O2", "-ffreestanding",
            "-ffunction-sections", "-fdata-sections", "-nostdlib", "-fno-builtin", f"-DSTACK_TOP={ram_bytes}",
            f"-Wl,-T,{FIRMWARE / 'link.ld'}", "-Wl,--no-relax", "-Wl,--gc-sections", "-fuse-ld=lld",
            str(FIRMWARE / "crt0.S"), str(source), str(FIRMWARE / "runtime.c"), "-o", str(elf),
        ], check=True)
        subprocess.run(["llvm-objcopy", "-O", "binary", str(elf), str(binary)], check=True)
        program = words(binary.read_bytes())
    if len(program) * 4 > ram_bytes:
        raise ValueError(f"{source.name} is {len(program) * 4} bytes, RAM is {ram_bytes}")
    return program


def assemble(source: str, origin: int) -> list[int]:
    with tempfile.TemporaryDirectory() as td:
        asm, elf, binary = Path(td) / "a.S", Path(td) / "a.elf", Path(td) / "a.bin"
        asm.write_text(source)
        subprocess.run([
            "clang", "--target=riscv32", "-march=rv32i", "-mabi=ilp32", "-nostdlib", f"-Wl,-Ttext={origin:#x}",
            "-Wl,--no-relax", "-Wl,-e,entry", "-fuse-ld=lld", str(asm), "-o", str(elf),
        ], check=True)
        subprocess.run(["llvm-objcopy", "-O", "binary", str(elf), str(binary)], check=True)
        return words(binary.read_bytes())
