data := "data/mnist"
mirror := "https://storage.googleapis.com/cvdf-datasets/mnist"
fashion := "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
# SLAYER's 1,100-sample slice of N-MNIST, pinned to the commit that added it
nmnist_small := "https://github.com/bamsumit/slayerPytorch/raw/3626f6a772d5d07171c39ddbeb13b3e23c6c8415/example/02_NMNIST_MLP/NMNISTsmall.zip"

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

# fetch the MNIST idx files, skipping ones already present, and check their sums
mnist: (_idx data mirror "mnist.sha256")

# Zalando's Fashion-MNIST in the same files; train on it with --mnist data/fashion-mnist
fashion-mnist: (_idx "data/fashion-mnist" fashion "fashion-mnist.sha256")

_idx dir url sums:
    mkdir -p {{dir}}
    for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do \
        [ -f {{dir}}/$f.gz ] || { curl -#fL --retry 3 -o {{dir}}/$f.gz.part {{url}}/$f.gz && mv {{dir}}/$f.gz.part {{dir}}/$f.gz; } || exit 1; \
    done
    cd {{dir}} && sha256sum --check --quiet "$OLDPWD/{{sums}}"

# 1,100 real N-MNIST recordings laid out for --nmnist data/nmnist-small
nmnist-small:
    mkdir -p data/nmnist-small
    [ -f data/nmnist-small/NMNISTsmall.zip ] || { curl -#fL --retry 3 -o data/nmnist-small/NMNISTsmall.zip.part {{nmnist_small}} && mv data/nmnist-small/NMNISTsmall.zip.part data/nmnist-small/NMNISTsmall.zip; }
    cd data/nmnist-small && sha256sum --check --quiet "$OLDPWD/nmnist-small.sha256"
    uv run --no-project python scripts/nmnist_small.py data/nmnist-small

# format, lint and test the way CI does
check:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace --exclude snn-live-py
    uv run --package snn-cuda pytest mnist-snn-cuda/tests scripts
    uv run --package sg-cuda pytest mnist-sg-cuda/tests

# rewrite the Rust sources in the standard format
fmt:
    cargo fmt --all

# build the snn_live Python module into the uv environment
snn-live-py:
    uv sync --package snn-live

# WTA STDP on CUDA (Python + CuPy), live view at http://127.0.0.1:7878
mnist-snn-cuda *args: mnist
    uv run --package snn-cuda --extra live python -m snn_cuda --mnist {{data}} {{args}}

# time every Rust model and train each over seeds (accuracy, spikes, energy); --robustness and --early-exit add curves, --save DIR keeps the build, --against DIR compares to one
bench-all *args: mnist
    uv run --no-project python scripts/bench_all.py --mnist {{data}} {{args}}

# surrogate-gradient SNN trained by backpropagation through time (CuPy, or NumPy without a GPU)
mnist-sg-cuda *args: mnist
    uv run --package sg-cuda python -m sg_cuda --mnist {{data}} {{args}}

# Rust CPU vs CUDA, interleaved best-of-5
snn-cuda-bench *args:
    cargo build --release -p mnist-snn-rs --examples
    uv run --package snn-cuda python -m snn_cuda.bench --mnist {{data}} {{args}}
