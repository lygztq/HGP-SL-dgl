import torch

def make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d+1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)

def threshold_and_support(input, dim=0):
    input_srt, _ = torch.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = make_ix_like(input, dim=dim)
    support = rhos * input_srt > input_cumsum
    print(support)

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size

input_size = (2, 3)
a = torch.randn(input_size)
print("Input: ", a)

print("Output: ", threshold_and_support(a, dim=1))
