data := "data/mnist"
mirror := "https://storage.googleapis.com/cvdf-datasets/mnist"

# train on MNIST and serve the live view at http://127.0.0.1:7878
mnist-snn-rs *args: mnist
    cargo run --release -p mnist-snn-rs -- --mnist {{data}} {{args}}

# fetch the MNIST idx files, skipping ones already present
mnist:
    mkdir -p {{data}}
    for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do \
        [ -f {{data}}/$f.gz ] || { curl -#fL --retry 3 -o {{data}}/$f.gz.part {{mirror}}/$f.gz && mv {{data}}/$f.gz.part {{data}}/$f.gz; } || exit 1; \
    done
