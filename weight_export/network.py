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
import argparse

##########
# IMPORTANT: If you want to use a different quantization bit width, you have to
# change the quantization strategies (Int8WeightPerTensorPoT and so on), as
# well as the bit width bw!
###########

parser = argparse.ArgumentParser(description="Train a quantised LeNet1-based CNN on the MNIST dataset and export its weights in weights_c1.h, weights_c3.h and weights_out.h.")
parser.add_argument("-b", "--bsize", type=int, default=64, help="Batch size")
parser.add_argument("-e", "--epochs", type=int, default=1, help="Epochs to train")
parser.add_argument("-s", "--seed", type=int, default=42, help="Set seed manually")
parser.add_argument("-c", "--report-scales", action='store_true', help="Print the activation scales")
parser.add_argument("-p", "--report-predictions", action='store_true', help="Print the predictions")
parser.add_argument("-a", "--report-accuracy", action='store_true', help="Print the test accuracy")
parser.add_argument("-q", "--suppress-progress", action='store_false', help="Suppress progress reporting")
parser.add_argument("-o", "--capture-outputs", type=int, default=-1, help="Capture intermediate outputs of all layers for a given input. -1 to capture no outputs.")
args = parser.parse_args()

class Int8WeightPerTensorPoT(Int8WeightPerTensorFloat):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
class Int8ActPerTensorPoT(Int8ActPerTensorFloat):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO

bsize = args.bsize
num_epochs = args.epochs
seed = args.seed
bw = 8

def parity_transform(img):
    arr = (np.array(img, dtype=np.uint8) // 2).astype(np.int8)
    return torch.tensor(arr, dtype=torch.int8).unsqueeze(0)

# Load the training dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True, 
    download=True,
    transform=parity_transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=parity_transform
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
torch.backends.cudnn.deterministic = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

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
            running_loss = 0.0
    if args.suppress_progress:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/100:.4f}')

model.eval()
correct = 0
total = 0

with torch.no_grad():
    capture_scales = args.report_scales
    output_predictions = args.report_predictions
    counter = 0
    breaker = args.capture_outputs
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        x = model[0](images)
        input_int = x.int().float()
        if capture_scales:
            scales.append(x.scale)
        x = model[1](x)
        weight_int = model[1].quant_weight().int().float()
        c1_out = F.conv2d(input_int, weight_int, bias=None, stride=1, padding=0)
        x = model[2](x)
        if counter == breaker:
            with open("c1_out.h", "w") as f:
                f.write(numpy_to_c_array(x.cpu().int().flatten().numpy(), "c1_out", c_type="int32_t"))
        if capture_scales:
            scales.append(x.scale)
        y = F.avg_pool2d(x.int().float(), 2)
        x = model[3](x)
        if counter == breaker:
            with open("s2_out.h", "w") as f:
                f.write(numpy_to_c_array(x.cpu().int().flatten().numpy(), "s2_out", c_type="int32_t"))
        x = model[4](x)
        x = model[5](x)
        if counter == breaker:
            with open("c3_out.h", "w") as f:
                f.write(numpy_to_c_array(x.cpu().int().flatten().numpy(), "c3_out", c_type="int32_t"))
        if capture_scales:
            scales.append(x.scale)
        x = model[6](x)
        if counter == breaker:
            with open("s4_out.h", "w") as f:
                f.write(numpy_to_c_array(x.cpu().int().flatten().numpy(), "s4_out", c_type="int32_t"))
        x = model[7](x)
        x = model[8](x)
        outputs = model[9](x)
        if counter == breaker:
            with open("fc_out.h", "w") as f:
                f.write(numpy_to_c_array(outputs.cpu().int().flatten().numpy(), "fc_out", c_type="int32_t"))
        if capture_scales:
            scales.append(outputs.scale)
            print(scales)
            capture_scales = False
            with open("act_scales.h", "w") as f:
                f.write("#include <stdint.h>\n")
                f.write(f"#define FIRST_ACT_SCALE 1 << {-torch.log2(scales[0]).int()}\n")
                f.write(f"#define SECOND_ACT_SCALE 1 << {-torch.log2(scales[1]).int()}\n")
                f.write(f"#define THIRD_ACT_SCALE 1 << {-torch.log2(scales[2]).int()}\n")
                f.write(f"#define FOURTH_ACT_SCALE 1 << {-torch.log2(scales[3]).int()}\n")
        _, predicted = torch.max(outputs.value, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if output_predictions:
            print(predicted.data.item())
        counter += 1

if args.report_accuracy:
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')

def numpy_to_c_array(arr, var_name="my_array", c_type="param_t"):
    shape = 1
    for d in arr.shape:
        shape *= d
    body = np.array2string(arr, max_line_width=np.inf, separator=', ', threshold=np.inf)
    body = body.replace('[', '{').replace(']', '}')
    return f"const {c_type} {var_name.upper()}[{shape}] = {body};"

def export_weights(layer_index, layer_name):
    fname = f"{layer_name}.h"
    tensor = model[layer_index].quant_weight()
    with open(fname, "w") as f:
        f.write(
                numpy_to_c_array(tensor.int().cpu().numpy().flatten(),
                layer_name))
        f.write(f"param_t {layer_name.upper()}_SCALE = {(-torch.log2(tensor.scale)).int()};")

export_weights(1, "weights_c1")
export_weights(4, "weights_c3")
export_weights(8, "weights_out")
