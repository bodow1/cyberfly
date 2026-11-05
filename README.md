<p align="center">
  <img src="docs/banner.gif" alt="cyberfly" width="100%">
</p>

Cyberfly is a small computer built out of simulated neurons and placed inside a simulated fruit fly brain.

The brain is the FlyWire connectome, a map of all 138,639 neurons in an adult *Drosophila* brain and the synapses between them. Shiu et al. (2024) turned this map into a simulation in which every neuron is a leaky integrate-and-fire (LIF) neuron, which adds up its inputs, leaks back toward rest and fires a spike when its voltage crosses a threshold.

The computer is a 32-bit RISC-V processor called the FPU (Fly Processing Unit), with 1 KiB of RAM. It is compiled into 83,864 LIF neurons that use the same model and parameters as the fly's own. A small GPU that renders a 3D maze brings the total to 186,381 neurons. The FPU's output bits excite visual neurons in the fly brain, and descending neurons, which carry commands from the brain to the body, write its input bits. A pair of radio neurons lets a bootloader receive new programs, and a neuron in the fly's mushroom body learns from reward when to call the FPU. In the embodied version a simulated fly in FlyGym sees through its compound eyes and is steered by its descending neurons. Programs are written in C.

Inspired by [Qwenputer](https://github.com/divagr18/qwenputer).

## How it works

### A neuron that behaves like a NOR gate

Every Boolean function can be built from NOR gates alone (Sheffer, 1913), so the first thing a computer made of neurons needs is a neuron that acts as a NOR gate. The idea that a threshold neuron can compute logic goes back to McCulloch and Pitts (1943), and networks of spiking neurons were later shown to be able to simulate Turing machines (Maass, 1996). Cyberfly applies this to the LIF model of Shiu et al. with its parameters unchanged.

A gate neuron has two kinds of input. A clock neuron excites it through a synapse of weight +242, and each logic input inhibits it through a synapse of weight −242. In this model one unit of synaptic weight moves the membrane voltage by 0.275 mV, so the clock synapse on its own is strong enough to make the gate fire. The clock neuron and the inputs fire on the phase before the gate's own phase. If none of the inputs fires, the clock spike pushes the gate over threshold and it fires, which the next gates read as a 1. If any input fires, its inhibition cancels the clock spike, the gate stays silent, and the next gates read a 0. A NOT gate is the same neuron with a single input.

<p align="center"><img src="docs/fig_nor.png" width="100%"></p>
<p align="center"><sub>Voltage of one NOR neuron in the simulator for the four input cases. Only A=0, B=0 reaches threshold.</sub></p>

### A clock

A gate fires only when its clock spike arrives, so the circuit needs a clock that tells each gate when to look at its inputs. The clock is a ring of neurons in which each neuron excites the next, so a single spike runs around the ring indefinitely. The time between one ring neuron and the next is one phase, which in this model is 49 time steps of 0.1 ms, or 4.9 ms.

<p align="center"><img src="docs/fig_clock.png" width="100%"></p>
<p align="center"><sub>A ring of 12 clock neurons in the simulator. The FPU uses a ring of 132.</sub></p>

Each gate is given a phase and takes its clock spike from the ring neuron one phase earlier. Its inputs have to fire one or two phases before it, so the compiler places every gate later in the ring than all the gates that feed it. The FPU needs 132 phases for its longest chain of gates plus the time to store the results, which makes one clock cycle 0.647 s. A ring neuron can drive at most 400 gates, so phases with more gates than that get extra copies of their clock neuron.

### Memory

A gate holds its value for only one phase, but a computer has to keep its registers and RAM from one cycle to the next, so each stored bit is a latch made of two neurons, A and B, that inhibit each other. This is the set-reset latch of digital electronics built from two NOR gates. A receives a clock spike on every even phase and B on every odd phase, and each fires on its clock spike unless the other fired on the phase before. Once A is firing, it silences B on every odd phase, B never gets to fire, and A keeps firing on every even phase. Once B is firing, the same happens the other way round. Exactly one of the two neurons therefore keeps firing, and which one it is holds the bit, with A firing for 1 and B firing for 0.

Writing a 1 takes a single spike from a set neuron that inhibits B. B misses one phase, A no longer gets silenced, and from then on A fires. Writing a 0 works the same way through a reset neuron that inhibits A. Reading needs no extra machinery, because any gate that needs the bit takes A as one of its inputs, or B when it needs the inverted bit.

<p align="center"><img src="docs/fig_latch.png" width="100%"></p>
<p align="center"><sub>One latch in the simulator. B fires while the bit is 0, a set spike switches the firing to A, and a reset spike switches it back.</sub></p>

The FPU's 31 registers, its program counter and its other state use 1,154 of these latches, each with its own set and reset neurons that write the next value at a fixed phase of every cycle. The 1 KiB of RAM is 256 words of 32 bits, or 8,192 latches. Each RAM bit uses four neurons, A, B, a read gate that fires when its word is addressed and the bit is 1, and a write gate that sets the bit. Address-decoder gates turn the 8-bit address into one word line per word. On a read, the read gates of the addressed word drive the 32 data lines. On a write, a clear neuron for the addressed word first resets all 32 of its latches, and the write gates then set the bits that are 1.

### Compiling the CPU

With gates, a clock and memory, a whole processor can be compiled. The FPU is a RISC-V core for the RV32I instruction set, written in about 160 lines of Verilog, a language for describing digital circuits. Yosys, an open-source synthesis tool, turns the Verilog into a netlist of 16,215 NOR and NOT gates and 1,154 flip-flops. The Cyberfly compiler then assigns each gate a phase after all of its inputs, creates one neuron per gate with its clock and input synapses, creates an A/B pair for each flip-flop with set and reset neurons that write the next value at a fixed phase in each cycle, and builds the RAM. A program is loaded by starting each RAM bit in the A or the B state.

The result is 83,864 neurons. One instruction takes two clock cycles, about 1.29 s of simulated time, and programs are written in C and compiled with clang. The table below counts where the neurons go, as printed by `docs/budget.py`.

| Part | Neurons |
| --- | ---: |
| RAM bits, 8,192 × 4 (A, B, read gate, write gate) | 32,768 |
| Delay relays and fan-out copies | 26,981 |
| CPU logic gates, one per gate in the netlist | 16,311 |
| Registers and other flip-flops, with their set and reset neurons | 4,552 |
| RAM address decoding and write select | 2,856 |
| Clock ring, clock copies and oscillators | 299 |
| Input port | 97 |
| Total | 83,864 |

The RAM is the largest part, at 35,624 neurons. The CPU core, meaning its gates, delay relays and flip-flops, is 47,844 neurons. Delay relays are needed because a gate's inputs must fire one or two phases before it, so a signal that is used many phases after it was computed is passed along a chain of NOT neurons until its consumer's phase.

### The GPU

The same compiler builds a small GPU next to the CPU, which adds 102,517 neurons. It renders a first-person view of a 16 × 16 maze. The FPU writes a camera position and angle to it, and for each of 32 screen columns the GPU steps a ray through the maze until it hits a wall, all 32 columns in parallel, and stores the distance. It also marks the columns where an enemy is drawn. The shooter program walks forward while the centre column's distance is more than 6 steps, turns toward the side whose DNa02 fired more, and shoots when DNp01 fires.

### Wiring it into the fly

<p align="center"><img src="docs/fig_system.png" width="100%"></p>
<p align="center"><sub>The FPU and GPU run in the same simulation as the fly brain and connect to it through LC10c, LC4, DNa02 and DNp01.</sub></p>

The FPU and GPU neurons are added to the fly's 138,639 and simulated together with them. Each output bit drives a few follower neurons, which fire every other phase while the bit is 1 and excite a group of LC10c and LC4 neurons, two types of visual projection neurons. In the shooter, the 32 enemy columns drive 32 such groups, with left columns going to the left side of the brain and right columns to the right.

From there the signal travels through the fly's own synapses to DNa02 and DNp01, two types of descending neurons, which carry commands from the brain to the body. DNa02 steers walking to one side, and DNp01, the giant fibre, triggers escape. Their spikes excite capture neurons that set bits 0 to 2 of the FPU's input port, and the program reads them with an ordinary load instruction.

In the embodied run, FlyGym renders what the fly sees on the 2 × 721 ommatidia of its compound eyes. Each ommatidium excites two LC10c neurons on its side, and sometimes an LC4 neuron, at a rate set by how dark it sees, and the difference between left and right DNa02 firing sets how the simulated body turns.

### The radio

The radio lets the FPU exchange messages with the outside world through one transmit neuron and one receive neuron, using 243 extra neurons in total. To send, a program stores a 14-bit value with bit 31 set, and in the next cycle 15 gated neurons fire a start spike and then one spike for each 1 bit onto the antenna neuron, 6 phases apart. On the receiving side a chain of relay neurons samples the receive neuron in each 6-phase slot, writes the bits into flip-flops and sets a valid bit, and the program clears the frame by storing bit 30.

`firmware/boot.c` walks a short zigzag, sends `?` and waits. The host decodes the `?` from the antenna neuron's spikes, asks a language model for a walking program, compiles it and sends it back as spikes on the receive neuron, as a header, three frames per 32-bit word and a checksum. The bootloader writes the words into RAM at address 0x200 and jumps there. A 36-word program takes 110 frames and 5,528 cycles to load.

### Learning to call the computer

The FPU is useful to the fly only if something in the brain decides when to use it. Cyberfly uses MBON09, an output neuron of the mushroom body, the part of the fly brain that forms learned associations. In the steering wiring, MBON09 spikes set input bit 3, so MBON09 firing acts as a call to the FPU.

Two sets of synapses onto MBON09 can change. The first are its synapses from Kenyon cells, the main neurons of the mushroom body. The second come from a small sign reader, a layer of neurons whose weights were trained offline to classify FlyGym renders of signs. Pixel neurons for the ommatidia drive both the reader and the visual neurons that feed the Kenyon cells.

In each trial the fly sees one sign for 0.5 s. If MBON09 fires, every changeable synapse onto it is updated by

```
dw = 0.05 · gain · (reward − 0.3 − 0.2) · min(pre spikes, 10) · min(MBON09 spikes, 3)
```

where the reward is 1 for an arrow sign and 0 for a stone, a blank board or nothing. Kenyon cell synapses can only weaken, reader synapses have a gain of 40, and all weights are bounded. A synapse therefore changes only when its input fired, MBON09 fired and a reward signal arrived, which is the three-factor form that dopamine plasticity takes in the mushroom body (Aso and Rubin, 2016). Before training, MBON09 fires for every kind of sign, and during training it also receives spontaneous spikes at 2 Hz. In a run of `examples/learn.py`, the share of held-out frames on which MBON09 fired went from 1.0 for every kind of sign before training to 1.0 for arrows, 0.4 for stones, 0.1 for blank boards and 0.0 for no sign after 60 trials.

### Testing

The tests run the same programs on three models and compare them cycle by cycle, an instruction-set simulator written in Python, the gate netlist and the spiking neurons. They also run the RISC-V rv32ui conformance suite on the netlist, load a program through the bootloader, echo a frame through the radio neurons, and train a small call circuit with sign-dependent and with shuffled reward.

## Quickstart

You need [uv](https://docs.astral.sh/uv/) and clang/lld with the RISC-V target.

```bash
git clone https://github.com/bodow1/cyberfly && cd cyberfly
uv sync
uv run pytest
```

The tests need no connectome data. To include the rv32ui suite, run

```bash
git clone https://github.com/riscv-software-src/riscv-tests
RISCV_TESTS=$PWD/riscv-tests uv run pytest tests/test_fpu.py::test_rv32ui
```

The examples need the connectome, about 420 MB, which is downloaded into `~/.cache/cyberfly/data` or `$CYBERFLY_DATA`.

```bash
uv run python -m cyberfly.connectome
```

The shooter runs in the brain without a body and checks every cycle against the gate model.

```bash
uv run python examples/shooter.py --cycles 200
```

The zigzag walk runs in FlyGym.

```bash
uv sync --extra body
uv run python examples/walk.py --seconds 20
```

The radio example asks a language model for a program and loads it through the bootloader. Set `CYBERFLY_LLM_BASE_URL`, `CYBERFLY_LLM_API_KEY` and `CYBERFLY_LLM_MODEL` for any OpenAI-compatible endpoint, or pass `--offline` for a fixed program.

```bash
uv run python examples/radio.py --request "walk a square, then rest"
```

The learning example trains MBON09 and plots its call rate.

```bash
uv run python examples/learn.py --trials 240
```

Pass `--gpu` to the shooter, walk and radio examples to simulate on CUDA (`uv sync --extra gpu`).

## What's real and what we added

| Real | Added |
| --- | --- |
| FlyWire v783 connectome with 138,639 neurons and their synapses | FPU and GPU neurons, 83,864 to 186,381 of them |
| Shiu et al. LIF model and its parameters | Clock ring, NOR-by-inhibition gates and latch pairs |
| Cell types and positions from the FlyWire annotations | Follower synapses from the FPU onto LC10c and LC4 |
| Paths from LC10c and LC4 through the brain to DNa02 and DNp01 | Synapses from DNa02, DNp01 and MBON09 onto FPU capture neurons |
| Kenyon cells, MBON09 and their synapses | Pixel neurons onto mushroom body visual inputs, the sign reader, and the three-factor rule on synapses onto MBON09 |
| FlyGym body, physics and compound-eye rendering | Random ommatidium-to-LC wiring and the DNa02-to-turn mapping |
| | Radio transmit and receive neurons |

## Layout

```
hw/          fpu.v (RV32I core), gpu.v (raycaster), top.v, neural.v
cyberfly/    hdl, logic, neurons, compiler, lif.c, lif, lif_gpu, runtime, iss, raycaster,
             connectome, brain, body, radio, llm, learning, firmware
firmware/    crt0, runtime, radio.h, boot.c, echo.c, shooter.c, fib.c, zigzag.c
examples/    shooter.py, walk.py, radio.py, learn.py
data/        signs.npz (FlyGym renders of signs, used by learn.py)
docs/        banner.py, figures.py
tests/       neurons, fpu, gpu, radio, learning
```

## Citing

Please cite Cyberfly as

```bibtex
@software{cyberfly2026,
  author  = {Wirth, Bodo and Kannan, Naveen},
  title   = {Cyberfly: a RISC-V computer in a fruit fly brain},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/bodow1/cyberfly}
}
```

Cyberfly builds on the following work.

- Shiu, P. K. et al. A *Drosophila* computational brain model reveals sensorimotor processing. *Nature* 634, 210–219 (2024).
- Dorkenwald, S. et al. Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138 (2024).
- Schlegel, P. et al. Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634, 139–152 (2024).
- Wang-Chen, S. et al. NeuroMechFly v2: simulating embodied sensorimotor control in adult *Drosophila*. *Nature Methods* 21, 2353–2362 (2024).
- Aso, Y. & Rubin, G. M. Dopaminergic neurons write and update memories with cell-type-specific rules. *eLife* 5, e16135 (2016).
- Hige, T., Aso, Y., Modi, M. N., Rubin, G. M. & Turner, G. C. Heterosynaptic plasticity underlies aversive olfactory learning in *Drosophila*. *Neuron* 88, 985–998 (2015).
- McCulloch, W. S. & Pitts, W. A logical calculus of the ideas immanent in nervous activity. *Bulletin of Mathematical Biophysics* 5, 115–133 (1943).
- Sheffer, H. M. A set of five independent postulates for Boolean algebras, with application to logical constants. *Transactions of the American Mathematical Society* 14, 481–488 (1913).
- Maass, W. Lower bounds for the computational power of networks of spiking neurons. *Neural Computation* 8, 1–40 (1996).
- Li, F. et al. The connectome of the adult *Drosophila* mushroom body provides insights into function. *eLife* 9, e62576 (2020).
- Agrawal, D. Qwenputer: a computer compiled into the weights of a pretrained language model (2026). https://github.com/divagr18/qwenputer

## License

Cyberfly is released under the MIT License, see [LICENSE](LICENSE). The FlyWire connectome is not part of this repository. `cyberfly.connectome` downloads it, and its use is governed by the [FlyWire terms](https://codex.flywire.ai/about_flywire).
