# neural_network.py  –  MLP dùng cho cá thể GA trong Chrome Dino
# Kiến trúc: Input(13) → Dense(256) → ReLU → Dense(128) → ReLU → Dense(3) → Softmax

import numpy as np
import os
import pickle
from typing import Optional


class NNConfig:
    INPUT_SIZE   = 15
    HIDDEN_SIZES = [256, 128]
    OUTPUT_SIZE  = 3


def _xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


class DenseLayer:
    def __init__(self, input_size: int, output_size: int):
        self.input_size  = input_size
        self.output_size = output_size
        self.weights = _xavier_init(input_size, output_size)
        self.bias    = np.zeros(output_size)

    @property
    def num_params(self) -> int:
        return self.weights.size + self.bias.size

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.weights) + self.bias

    def set_weights(self, flat: np.ndarray, start: int) -> int:
        w_size = self.weights.size
        b_size = self.bias.size
        self.weights = flat[start:start + w_size].reshape(self.weights.shape)
        self.bias    = flat[start + w_size:start + w_size + b_size]
        return start + w_size + b_size

    def get_weights(self, flat: list) -> list:
        flat.extend(self.weights.flatten())
        flat.extend(self.bias.flatten())
        return flat


class ReLULayer:
    def __init__(self):
        self.cache = None

    @property
    def num_params(self) -> int:
        return 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x.copy()
        return np.maximum(0, x)

    def set_weights(self, flat: np.ndarray, start: int) -> int:
        return start

    def get_weights(self, flat: list) -> list:
        return flat


class SoftmaxLayer:
    def __init__(self):
        self.cache = None

    @property
    def num_params(self) -> int:
        return 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x     = np.exp(x_shifted)
        out       = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        self.cache = out
        return out

    def set_weights(self, flat: np.ndarray, start: int) -> int:
        return start

    def get_weights(self, flat: list) -> list:
        return flat


class DinoNet:
    def __init__(self, cfg: Optional[NNConfig] = None):
        self.cfg = cfg or NNConfig()
        self._build_network()

    def _build_network(self):
        self.layers = [
            DenseLayer(self.cfg.INPUT_SIZE,  self.cfg.HIDDEN_SIZES[0]),
            ReLULayer(),
            DenseLayer(self.cfg.HIDDEN_SIZES[0], self.cfg.HIDDEN_SIZES[1]),
            ReLULayer(),
            DenseLayer(self.cfg.HIDDEN_SIZES[1], self.cfg.OUTPUT_SIZE),
        ]

    def forward(self, state: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            x = state.reshape(1, -1)
        else:
            x = state
        if x.shape[-1] > self.cfg.INPUT_SIZE:
            x = x[..., :self.cfg.INPUT_SIZE]
        for layer in self.layers:
            x = layer.forward(x)
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        if state.ndim == 1:
            return probs.flatten()
        return probs

    def predict(self, state: np.ndarray) -> int:
        return int(np.argmax(self.forward(state)))

    @property
    def num_params(self) -> int:
        return sum(layer.num_params for layer in self.layers)

    def get_flat_weights(self) -> np.ndarray:
        flat_list = []
        for layer in self.layers:
            layer.get_weights(flat_list)
        return np.array(flat_list, dtype=np.float64)

    def set_flat_weights(self, flat_weights: np.ndarray) -> None:
        flat = np.asarray(flat_weights, dtype=np.float64)
        pos  = 0
        for layer in self.layers:
            pos = layer.set_weights(flat, pos)

    def randomize(self) -> None:
        self._build_network()

    def copy(self) -> "DinoNet":
        new_net = DinoNet(self.cfg)
        new_net.set_flat_weights(self.get_flat_weights())
        return new_net

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "weights": self.get_flat_weights(),
                "nn_config": {
                    "input_size":   self.cfg.INPUT_SIZE,
                    "hidden_sizes": self.cfg.HIDDEN_SIZES,
                    "output_size":  self.cfg.OUTPUT_SIZE,
                },
            }, f)

    @classmethod
    def load(cls, path: str) -> "DinoNet":
        with open(path, "rb") as f:
            data = pickle.load(f)
        cfg = NNConfig()
        cfg.INPUT_SIZE   = data["nn_config"]["input_size"]
        cfg.HIDDEN_SIZES = data["nn_config"]["hidden_sizes"]
        cfg.OUTPUT_SIZE  = data["nn_config"]["output_size"]
        net = cls(cfg)
        net.set_flat_weights(data["weights"])
        return net

    def __repr__(self) -> str:
        arch = f"{self.cfg.INPUT_SIZE}"
        for h in self.cfg.HIDDEN_SIZES:
            arch += f" → {h}"
        arch += f" → {self.cfg.OUTPUT_SIZE}"
        return f"<DinoNet {arch}  params={self.num_params}>"
