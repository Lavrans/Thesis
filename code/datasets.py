from numpy.typing import NDArray
from torch import Tensor
from torch import float32
from torch import tensor
from torch.types import Device
from torch.utils.data import Dataset


class LSTMDataset(Dataset):  # type: ignore
    def __init__(self, samples: NDArray, underlying: NDArray, targets: NDArray, device: Device) -> None:
        self.samples = tensor(samples, dtype=float32, device=device)
        self.underlying = tensor(underlying, dtype=float32, device=device)
        self.targets = tensor(targets, dtype=float32, device=device)
        super().__init__()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.underlying[index],
            self.samples[index],
            self.targets[index],
        )


class MLPDataset(Dataset):  # type: ignore
    def __init__(self, samples: NDArray, targets: NDArray, device: Device) -> None:
        self.samples = tensor(samples, dtype=float32, device=device)
        self.targets = tensor(targets, dtype=float32, device=device)
        super().__init__()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return (
            self.samples[index],
            self.targets[index],
        )
