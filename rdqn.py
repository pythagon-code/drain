import torch
from torch import Tensor, nn

HiddenState = tuple[Tensor, Tensor]

class RDQN(nn.Module):
    @staticmethod
    def _make_layer(in_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(),
        )

    def __init__(self, state_dim: int, action_dim: int, num_rnn_layers: int, hidden_dim: int) -> None:
        super().__init__()
        self._stem = nn.Sequential(
            self._make_layer(state_dim, hidden_dim),
            *(self._make_layer(hidden_dim, hidden_dim) for _ in range(2)),
        )
        self._rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
        )
        self._head = nn.Sequential(
            *(self._make_layer(hidden_dim, hidden_dim) for _ in range(2)),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        states: Tensor,
        hidden: HiddenState,
        integrity: Tensor | None
    ) -> tuple[Tensor, HiddenState, Tensor | None]:
        trunc_length = states.size(1)
        stem_outs = self._stem(states)
        hs, cs = (x.unsqueeze(1).repeat(1, trunc_length + 1, 1, 1) for x in hidden)
        for i in range(1, trunc_length+1):
            hs[:, i, :, :], cs[:, i, :, :] = (
                x.permute(1, 0, 2)
                for x in self._rnn(
                    stem_outs[:, i - 1:i, :],
                    (
                        hs[:, i - 1, :, :].permute(1, 0, 2).contiguous(),
                        cs[:, i - 1, :, :].permute(1, 0, 2).contiguous(),
                    )
                )[1]
            )
        if integrity is not None:
            integrities = integrity.unsqueeze(1).repeat(1, trunc_length + 1, 1)
            integrities = integrities + torch.arange(
                trunc_length + 1,
                dtype=states.dtype,
                device=states.device,
            ).view(1, -1, 1)
        qs = self._head(hs[:, 1:, -1, :])
        return qs, (hs[:, 1:, :, :], cs[:, 1:, :, :]), integrities[:, 1:, :] if integrity is not None else None
    
    def simple_forward(self, state: Tensor, hidden: HiddenState) -> tuple[Tensor, HiddenState]:
        stem_out = self._stem(state)
        rnn_outs, hidden = self._rnn(stem_out, tuple((x.permute(1, 0, 2) for x in hidden)))
        hidden = tuple((x.permute(1, 0, 2) for x in hidden))
        head_out = self._head(rnn_outs)
        return head_out, hidden