import pytest

from cyberfly.firmware import build
from cyberfly.hdl import Netlist


@pytest.fixture(scope="session")
def fpu_netlist():
    return Netlist.build("fly_fpu")


@pytest.fixture(scope="session")
def gpu_netlist():
    return Netlist.build("fly_fpu_gpu")


@pytest.fixture(scope="session")
def fib():
    return build("fib.c")


@pytest.fixture(scope="session")
def shooter():
    return build("shooter.c")
