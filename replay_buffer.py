import torch
from torch import Tensor

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        trunc_length: int,
        item_size: int,
        batch_size: int,
        dtype: torch.dtype,
        device: str
    ) -> None:
        self._start = self._size = 0
        self._capacity = capacity
        self._trunc_length = trunc_length
        self._item_size = item_size
        self._batch_size = batch_size
        self._device = device
        self._buffer = torch.zeros(capacity, item_size, dtype=dtype, device=device)

    def get_size(self) -> int:
        return self._size
    
    def add(self, item) -> None:
        self._buffer[(self._start + self._size) % self._capacity] = item
        if self._size == self._capacity:
            self._start = (self._start + 1) % self._capacity
        else:
            self._size += 1

    def sample(self) -> tuple[Tensor, Tensor]:
        assert self._size >= self._batch_size + self._trunc_length
        max_jump = self._size - self._trunc_length - 1
        start_idx = self._start + torch.randint(max_jump, (self._batch_size,), device=self._buffer.device)
        start_idx = start_idx.unsqueeze(1).expand(-1, self._trunc_length)
        offsets = torch.arange(self._trunc_length, device=self._buffer.device).unsqueeze(0).expand(self._batch_size, -1)
        idxs = (start_idx + offsets) % self._capacity
        sample = self._buffer[idxs]
        return sample, idxs

    def realign_hidden_states(
        self,
        idxs: Tensor,
        hs: Tensor,
        cs: Tensor,
        next_hs: Tensor,
        next_cs: Tensor,
        integrities: Tensor,
    ) -> None:
        hidden_size = hs.size(2) * hs.size(3)
        integrity_start = self._item_size - 1
        next_cs_start = integrity_start - hidden_size
        next_hs_start = next_cs_start - hidden_size
        cs_start = next_hs_start - hidden_size
        hs_start = cs_start - hidden_size
        batch = self._buffer[idxs]
        flat_hs = hs.reshape(*hs.shape[:2], hidden_size)
        flat_cs = cs.reshape(*cs.shape[:2], hidden_size)
        flat_next_hs = next_hs.reshape(*next_hs.shape[:2], hidden_size)
        flat_next_cs = next_cs.reshape(*next_cs.shape[:2], hidden_size)
        current_integrities = batch[:, :, integrity_start:integrity_start + 1]
        should_realign = integrities > current_integrities
        batch[:, :, hs_start:cs_start] = torch.where(
            should_realign.expand_as(flat_hs),
            flat_hs,
            batch[:, :, hs_start:cs_start],
        )
        batch[:, :, cs_start:next_hs_start] = torch.where(
            should_realign.expand_as(flat_cs),
            flat_cs,
            batch[:, :, cs_start:next_hs_start],
        )
        batch[:, :, next_hs_start:next_cs_start] = torch.where(
            should_realign.expand_as(flat_next_hs),
            flat_next_hs,
            batch[:, :, next_hs_start:next_cs_start],
        )
        batch[:, :, next_cs_start:integrity_start] = torch.where(
            should_realign.expand_as(flat_next_cs),
            flat_next_cs,
            batch[:, :, next_cs_start:integrity_start],
        )
        batch[:, :, integrity_start:integrity_start + 1] = torch.where(
            should_realign,
            integrities,
            current_integrities,
        )
        self._buffer[idxs] = batch