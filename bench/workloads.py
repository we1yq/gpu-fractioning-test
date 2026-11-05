"""Synthetic GPU workloads.

Each workload isolates one source of contention so results can be attributed:

  gemm         SM contention        (large dense matrix multiply)
  membw        memory bandwidth     (STREAM-style copy + triad on large tensors)
  smallkernel  scheduling overhead  (tiny kernels, one per step, high frequency)
  hostcopy     PCIe                 (host <-> device copies, no compute)

A workload allocates everything up front, so an out-of-memory condition shows up
during setup rather than in the middle of the measured window. `step()` submits
one unit of work; the caller is responsible for synchronizing and timing it.
"""

import torch


class Workload:
    def __init__(self, size, device, seed):
        self.size = size
        self.device = device
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def _randn(self, *shape):
        # Generate on the CPU so every backend gets identical input for a seed.
        return torch.randn(*shape, generator=self.generator, dtype=torch.float32).to(self.device)

    def step(self):
        raise NotImplementedError


class Gemm(Workload):
    """size = matrix edge length. Holds three size x size float32 matrices."""

    default_size = 4096

    def __init__(self, size, device, seed):
        super().__init__(size, device, seed)
        self.a = self._randn(size, size)
        self.b = self._randn(size, size)
        self.out = torch.empty_like(self.a)

    def step(self):
        torch.mm(self.a, self.b, out=self.out)


class MemBw(Workload):
    """size = MiB per array. Holds three arrays, so the footprint is 3 x size."""

    default_size = 680
    scalar = 3.0

    def __init__(self, size, device, seed):
        super().__init__(size, device, seed)
        elements = size * 1024 * 1024 // 4
        self.a = torch.empty(elements, dtype=torch.float32, device=device)
        self.b = self._randn(elements)
        self.c = self._randn(elements)

    def step(self):
        self.a.copy_(self.b)
        torch.add(self.b, self.c, alpha=self.scalar, out=self.a)


class SmallKernel(Workload):
    """size = matrix edge length. One small GEMM per step, mimicking an inference call."""

    default_size = 256

    def __init__(self, size, device, seed):
        super().__init__(size, device, seed)
        self.a = self._randn(size, size)
        self.b = self._randn(size, size)
        self.out = torch.empty_like(self.a)

    def step(self):
        torch.mm(self.a, self.b, out=self.out)


class HostCopy(Workload):
    """size = MiB per direction. The only workload that crosses PCIe for its data."""

    default_size = 256

    def __init__(self, size, device, seed):
        super().__init__(size, device, seed)
        elements = size * 1024 * 1024 // 4
        self.device_buffer = torch.empty(elements, dtype=torch.float32, device=device)
        self.host_buffer = torch.randn(elements, generator=self.generator, dtype=torch.float32)
        if str(device).startswith("cuda"):
            self.host_buffer = self.host_buffer.pin_memory()

    def step(self):
        self.device_buffer.copy_(self.host_buffer)
        self.host_buffer.copy_(self.device_buffer)


WORKLOADS = {"gemm": Gemm, "hostcopy": HostCopy, "membw": MemBw, "smallkernel": SmallKernel}
