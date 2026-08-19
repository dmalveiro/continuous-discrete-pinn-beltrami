import torch
import param
from param import precision, device

def pytorch_output(var1):

    # Convert input 'var1' to torch tensor
    if not isinstance(var1, torch.Tensor):
        var1 = torch.as_tensor(var1, dtype=precision, device=device)

    return var1
