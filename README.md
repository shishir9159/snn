# Spiking Neural Network
Implementing different algorithms for learning and purpose of iterative improvement and comparisons(benchmarks). Model training on MNIST(and synthetic) is streamed(state) to a live dashboard to provide better intuition about the learning.

## Models
| Project | Model | MNIST test accuracy |
|---|---|---|
| `mnist-snn-rs` | Winner-takes-all layer with lateral inhibition, pair STDP, weight normalisation and homeostasis | ~71% on 6k images (40 outputs), ~75% on 20k (100 outputs) |
| `mnist-spncn-rs` | Spiking predictive coding (SpNCN), 400→100 latent layers | 89.9% after 1 online epoch |
| `mnist-pcsnn-rs` | Predictive-coding spike timing (PC-SNN) | ~90–92% after 5–8 epochs |
| `mnist-pcl-rs` | Predictive coding light (PCL) on saccade events | 82.5% after 3k images |
| `mnist-snn-cuda` | The WTA model on CUDA (Python + CuPy), bit-identical to `mnist-snn-rs` | same as above |

The predictive-coding papers needed changes to train at all; they are noted in the code next to each change. PCL's plastic inhibition roughly halves the spikes against `--no-inhibition`.

## Quick start
```sh
just mnist
just mnist-snn-rs
just mnist-spncn-rs
just mnist-pcsnn-rs
just mnist-pcl-rs
```

## Live view
- **WTA:** weight maps, spike raster, membrane potential, class histogram. Click a neuron to pin it.
- **SpNCN, PC-SNN, PCL:** population activity and receptive fields. Click a unit to probe it, Esc to clear.

The pause and speed controls steer training (0.2× to max). Joining late is fine: a new viewer gets a snapshot of the run so far.

## Flags
Shared by every Rust model(todo):

| Flag | Meaning |
|---|---|
| `--mnist DIR` / `--png DIR` | MNIST IDX files, or `DIR/<class>/*.png`; default is synthetic digits |
| `--train N` / `--test N` | number of training / test images |
| `--epochs N` | passes over the training set (default 1 headless, 10 live) |
| `--seed N` | random seed (default 1); runs are reproducible |
| `--headless` | no live view, print progress and exit |
| `--port N` | live view port (default 7878) |
| `--speed X` | multiple of real time at 1 ms per step, or `max` (default 1) |

Per model: `mnist-snn-rs` has `--outputs N`, `--load FILE`, `--save FILE`, `--weights-png FILE`; `mnist-spncn-rs` has `--layers N,N`, `--dt MS`, `--fields-png FILE`; `mnist-pcsnn-rs` has `--hidden N`, `--fields-png FILE`; `mnist-pcl-rs` has `--no-inhibition`. `--help` lists them.

## CUDA

```sh
just mnist-snn-cuda --outputs 100
just snn-cuda-bench
uv run --package snn-cuda python -m snn_cuda.profile --mnist data/mnist
```

Python models can drive the same dashboard through `snn_live`: `just snn-live-py` builds the module.

## Profiling
```sh
cargo run --release -p mnist-snn-rs --example profile -- data/mnist 20000
cargo run --release -p mnist-spncn-rs --example spncn-profile -- data/mnist
```

## Tests
```sh
cargo test --workspace
uv run --package snn-cuda pytest
```

## Layout
```
crates/snn-core      rng, datasets, encoding, events, readout, weights, run control
crates/snn-live      frame protocol, hub, HTTP/WebSocket server, shared CLI
crates/snn-live-py   PyO3 module exposing the live view to Python
mnist-*-rs           the four Rust models, each with trainer, tap and profile example
mnist-snn-cuda       CUDA port of the WTA model
web/                 dashboard: index.html plus one script per model in views/
```

## Goal
- [ ] ~~ratatui~~
- [x] documentation
