import torch
import numpy as np

data = [[1,2],[3,4]]    
x_data = torch.tensor(data)
print(x_data)

x_ones = torch.ones_like(x_data)
print(x_ones)

x_rand = torch.rand_like(x_data, dtype = torch.float)    
print(x_rand)

tensor = torch.rand(3,4)
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
print(tensor.device)

tensor = torch.ones(4,4)
print(tensor[0])

print(tensor.de