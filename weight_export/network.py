import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import brevitas.nn as qnn
from brevitas.quant.scaled_int import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat
)
from brevitas.core.restrict_val import RestrictValueType

class Int8WeightPerTensorPoT(Int8WeightPerTensorFloat):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
class Int8ActPerTensorPoT(Int8ActPerTensorFloat):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO

bsize = 64
num_epochs = 1
seed = 42
bw = 8

transform = transforms.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.int8)
])

# Load the training dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True, 
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


model = nn.Sequential(
                qnn.QuantIdentity(act_quant=Int8ActPerTensorPoT, bit_width=bw, return_quant_tensor=True),
                qnn.QuantConv2d(1, 4, 5, bias=False, weight_bit_width=bw, weight_quant=Int8WeightPerTensorPoT),
                qnn.QuantReLU(act_quant=Int8ActPerTensorPoT, bit_width=bw, return_quant_tensor=True),
                qnn.TruncAvgPool2d(2),
                qnn.QuantConv2d(4, 12, 5, bias=False, weight_bit_width=bw, weight_quant=Int8WeightPerTensorPoT),
                qnn.QuantReLU(act_quant=Int8ActPerTensorPoT, bit_width=bw, return_quant_tensor=True),
                qnn.TruncAvgPool2d(2),
                nn.Flatten(),
                qnn.QuantLinear(4 * 4 * 12, 10, weight_bit_width=bw, bias=False, weight_quant=Int8WeightPerTensorPoT),
                qnn.QuantIdentity(act_quant=Int8ActPerTensorPoT, bit_width=bw, return_quant_tensor=True)
        )

model.train()

torch.manual_seed(seed)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Training loop
train_losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % 100 == 0:
            train_losses.append(running_loss/100)
            running_loss = 0.0
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/100:.4f}')

model.eval()
correct = 0
total = 0

def numpy_to_c_array(arr, var_name="my_array", c_type="param_t"):
    shape = 1
    for d in arr.shape:
        shape *= d
    body = np.array2string(arr, max_line_width=np.inf, separator=', ', threshold=np.inf)
    body = body.replace('[', '{').replace(']', '}')
    return f"const {c_type} {var_name.upper()}[{shape}] = {body};"

def export_tensor(qtensor, name):
    arr = qtensor.int().cpu().numpy().flatten()

    with open(f"{name}.h", "w") as f:
        f.write("#include <stdint.h>\n\n")
        f.write(numpy_to_c_array(arr, name, c_type="int32_t"))
        f.write(f"\n\n#define {name.upper()}_SIZE {arr.size}\n")

with torch.no_grad():
    y = True
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        x0 = model[0](images)
        x = model[1](x0)
        acc = F.conv2d(x0.int().float(), model[1].quant_weight().int().float(), bias=None)
        if y:
            export_tensor(acc, "c1_out")
            y = False
        x = model[2](x)
        #print(x.scale)
        x = model[3](x)
        x = model[4](x)
        x = model[5](x)
        #print(x.scale)
        x = model[6](x)
        x = model[7](x)
        x = model[8](x)
        outputs = model[9](x)
        #print(outputs.scale)
        _, predicted = torch.max(outputs.value, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(predicted.data.item())

accuracy = 100 * correct / total
print(f'Accuracy on the test set: {accuracy:.2f}%')

def export_weights(layer_index, layer_name):
    fname = f"{layer_name}.h"
    header = "#include <stdint.h>\n\n"
    tensor = model[layer_index].quant_weight()
    with open(fname, "w") as f:
        f.write(header)
        f.write(
                numpy_to_c_array(tensor.int().cpu().numpy().flatten(),
                layer_name))
        f.write(f"\nparam_t {layer_name.upper()}_SCALE = {(-torch.log2(tensor.scale)).int()};")

# Export weights
export_weights(1, "weights_c1")
export_weights(4, "weights_c3")
export_weights(8, "weights_out")
