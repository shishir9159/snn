data := "data/mnist"
mirror := "https://storage.googleapis.com/cvdf-datasets/mnist"

# train on MNIST and serve the live view at http://127.0.0.1:7878
mnist-snn-rs *args: mnist
    cargo run --release -p mnist-snn-rs -- --mnist {{data}} {{args}}

# spiking predictive coding (SpNCN) on MNIST, live view at http://127.0.0.1:7878
mnist-spncn-rs *args: mnist
    cargo run --release -p mnist-spncn-rs -- --mnist {{data}} {{args}}

# predictive-coding spike timing (PC-SNN) on MNIST, live view at http://127.0.0.1:7878
mnist-pcsnn-rs *args: mnist
    cargo run --release -p mnist-pcsnn-rs -- --mnist {{data}} {{args}}

# predictive coding light (PCL) on MNIST saccade events, live view at http://127.0.0.1:7878
mnist-pcl-rs *args: mnist
    cargo run --release -p mnist-pcl-rs -- --mnist {{data}} {{args}}

# fetch the MNIST idx files, skipping ones already present
mnist:
    mkdir -p {{data}}
    for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do \
        [ -f {{data}}/$f.gz ] || { curl -#fL --retry 3 -o {{data}}/$f.gz.part {{mirror}}/$f.gz && mv {{data}}/$f.gz.part {{data}}/$f.gz; } || exit 1; \
    done

# format, lint and test the way CI does
check:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace --exclude snn-live-py
    uv run --package snn-cuda pytest mnist-snn-cuda/tests scripts

# rewrite the Rust sources in the standard format
fmt:
    cargo fmt --all

# build the snn_live Python module into the uv environment
snn-live-py:
    uv sync --package snn-live

# WTA STDP on CUDA (Python + CuPy), live view at http://127.0.0.1:7878
mnist-snn-cuda *args: mnist
    uv run --package snn-cuda --extra live python -m snn_cuda --mnist {{data}} {{args}}

# time every Rust model and train each briefly (accuracy, spikes, synops); --save DIR keeps the build, --against DIR compares to one
bench-all *args: mnist
    uv run --no-project python scripts/bench_all.py --mnist {{data}} {{args}}

# Rust CPU vs CUDA, interleaved best-of-5
snn-cuda-bench *args:
    cargo build --release -p mnist-snn-rs --examples
    uv run --package snn-cuda python -m snn_cuda.bench --mnist {{data}} {{args}}
