"""Model constants; defaults match mnist-snn-rs, tuned on MNIST."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Lif:
    rest: float = 0.0
    # lowest reachable potential; inhibited neurons are pushed here
    floor: float = -500.0
    # subtracted every step while above rest
    leak: float = 0.15
    # steps a neuron stays silent after a spike
    t_ref: int = 30


@dataclass(frozen=True)
class Encode:
    # potential range mapped onto the frequency range (clamped)
    pot_min: float = -1.069
    pot_max: float = 2.781
    freq_min: float = 1.0
    freq_max: float = 20.0
    # spike period is ceil(period_scale / freq) steps
    period_scale: float = 600.0


@dataclass(frozen=True)
class Stdp:
    # depression when the pre spike follows the post spike
    a_plus: float = 0.8
    # potentiation when the pre spike leads
    a_minus: float = 0.3
    tau_plus: float = 8.0
    tau_minus: float = 5.0
    # window around a post spike, in steps
    t_back: int = 20
    t_fore: int = 20
    # learning rate
    sigma: float = 0.1
    w_min: float = 0.0
    w_max: float = 2.0


@dataclass(frozen=True)
class Params:
    # steps an image is presented for
    time_steps: int = 200
    lif: Lif = field(default_factory=Lif)
    encode: Encode = field(default_factory=Encode)
    stdp: Stdp = field(default_factory=Stdp)
    # per-image threshold = peak simultaneous input spikes / threshold_div
    threshold_div: float = 1.0
    # depression of the winner's synapses from inputs that never spiked
    silent_penalty: float = 0.02
    # L2 norm every weight row is rescaled to after learning (0 disables)
    w_norm: float = 8.0
    # homeostasis rate for thresholds
    theta_rate: float = 0.05
    # initial weights are uniform in this range
    w_init: tuple = (0.0, 0.4)
