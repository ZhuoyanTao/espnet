import torch
from hificodec_quantizer import HiFiCodecQuantizer   # your file

B, C, T = 2, 512, 64
x = torch.randn(B, C, T)                   # pretend encoder features

q = HiFiCodecQuantizer(
    dim=512, n_groups=8, n_codes=1024,
    codebook_loss_lambda=1.0,
    commitment_loss_lambda=0.25,
    residual_layers=2,
)

z_q, q_loss, indices = q(x)                # z_q: (B,C,T), q_loss: scalar
print(z_q.shape, q_loss.item())
# indices is list of length 16 (= 2 residual layers × 8 groups), each (B*T,)
