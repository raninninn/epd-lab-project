import os
import time
import math
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Hyperparameters
LEARNING_RATE = 0.01
PATIENCE = 3                # stop after this many epochs without validation loss improvement
MIN_DELTA = 0.001           # improvement below this doesn't count
MAX_TRAIN_TIME_SEC = 10 * 60
BATCH_SIZE = 64
VAL_FRAC = 0.1              # fraction of training data held out for validation

# Quantization
BITS = 8
QMAX = 2**(BITS - 1) - 1    # Largest quantized value
QMIN = -QMAX - 1            # Smallest quantized value


class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()

        self.model = nn.Sequential(OrderedDict([
            # Input 1 @ 28 x 28
            ("conv1", nn.Conv2d(1, 4, 5)),
            ("relu1", nn.ReLU()),
            # Feature map 4 @ 24 x 24
            ("pool1", nn.AvgPool2d(2, 2)),
            # Feature map 4 @ 12 x 12
            ("conv2", nn.Conv2d(4, 12, 5)),
            ("relu2", nn.ReLU()),
            # Feature map 12 @ 8 x 8
            ("pool2", nn.AvgPool2d(2, 2)),
            # Feature map 12 @ 4 x 4
            ("flatten", nn.Flatten()),
            # Tensor [12 * 4 * 4]
            ("fc", nn.Linear(12 * 4 * 4, 10)),
            # Output [10]
        ]))

    def forward(self, x):
        return self.model(x)


def train(model, train_loader, val_loader):
    # Cross-entropy: standard loss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    # Stochastic gradient descent
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    best_val, bad_epochs, epoch = float('inf'), 0, 0

    # Train until validation loss stops improving (or time cap)
    start = time.time()
    while bad_epochs < PATIENCE and time.time() - start < MAX_TRAIN_TIME_SEC:
        model.train()
        # Mini-batch SGD
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        # Average loss on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                val_loss += criterion(model(images), labels).item()
        val_loss /= len(val_loader)

        # Save best model
        if val_loss < best_val - MIN_DELTA:
            best_val, bad_epochs = val_loss, 0
            torch.save(model.state_dict(), "checkpoints/lenet1.pt")
        else:
            bad_epochs += 1
        epoch += 1
        print(f"Epoch {epoch}  val_loss: {val_loss:.4f}")

    # Restore best checkpoint so callers get the best model, not the last
    model.load_state_dict(torch.load("checkpoints/lenet1.pt"))


# Returns fraction of correctly classified samples
def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            predicted = model(images).argmax(dim=1)
            correct += (predicted == labels).sum().item()
    return correct / len(loader.dataset)


# Returns fraction of correctly classified samples for quantized weights and shifts
def evaluate_int8(weights, act_shifts, loader):
    # Accuracy = fraction of correctly classified samples
    correct = 0
    for images, labels in loader:
        logits = forward_int8(images, weights, act_shifts)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


# Find the largest scale factor 2^n that keeps all values within quantized integer range (max|t| * 2^n <= QMAX)
def choose_shift(t):
    return math.floor(math.log2(QMAX / t.abs().max()))


# Float -> int8: scale tensor and round (banker's rounding) to int
def quantize_tensor_int8(t, n):
    rounded = torch.round(t * 2**n)
    # Activations can exceed the calibrated range at runtime, cut off at quantized limits
    return torch.clamp(rounded, QMIN, QMAX).to(torch.int8)


def quantize_model(state_dict):
    quantized = {}
    for name, W in state_dict.items():
        n = choose_shift(W)
        quantized[name] = {"q": quantize_tensor_int8(W, n), "shift": n}
    return quantized


# Where the int pipeline requantizes back to int8: after each conv/fc.
# Conv outputs are measured after ReLU (negatives die anyway, only the
# positive max matters); fc logits are measured raw (both signs matter).
CALIB_POINTS = {"relu1": "conv1", "relu2": "conv2", "fc": "fc"}
CALIB_BATCHES = 5

def calibrate(model, loader):
    # Record the largest |activation| per requantization point on sample data
    maxes = {"input": 0.0, "conv1": 0.0, "conv2": 0.0, "fc": 0.0}
    model.eval()
    with torch.no_grad():
        for batch, (images, _) in enumerate(loader):
            if batch == CALIB_BATCHES:
                break
            x = images
            maxes["input"] = max(maxes["input"], x.abs().max().item())
            # Run layer by layer to see the intermediate outputs
            for name, layer in model.model.named_children():
                x = layer(x)
                if name in CALIB_POINTS:
                    point = CALIB_POINTS[name]
                    maxes[point] = max(maxes[point], x.abs().max().item())

    # Same rule as for weights: largest 2^n that keeps the observed max in range
    shifts = {point: math.floor(math.log2(QMAX / m)) for point, m in maxes.items()}
    return shifts


# Rescale accumulator from scale 2^shift_in back to int8 range at 2^shift_out
def requantize(acc, shift_in, shift_out):
    q = torch.round(acc / 2**(shift_in - shift_out))
    return torch.clamp(q, QMIN, QMAX)


def avgpool_int(x):
    return torch.round(F.avg_pool2d(x, 2))


def relu(x):
    return torch.clamp(x, min=0)


# Forward pass for quantized model, return logits
def forward_int8(images, weights, act_shifts):
    # Returns int8 weight + scale, int8 bias + scale for a layer
    def get(name):
        weight = weights[f"model.{name}.weight"]
        bias = weights[f"model.{name}.bias"]
        return weight["q"].double(), weight["shift"], bias["q"].double(), bias["shift"]

    # Bring bias from its own scale to the accumulator scale
    def align_bias(bias, shift_bias, shift_acc):
        return bias * 2 ** (shift_acc - shift_bias)

    # Quantize input layer
    shift_input = act_shifts["input"]
    x = quantize_tensor_int8(images, shift_input).double()

    # Conv1, Requantize, ReLU, AvgPool
    weight, shift_weight, bias, shift_bias = get("conv1")
    shift_acc = shift_input + shift_weight
    # Reshape bias to conv 4-D output: [channels] -> [1, channels, 1, 1]
    bias_per_channel = align_bias(bias, shift_bias, shift_acc).view(1, -1, 1, 1)
    acc = F.conv2d(x, weight) + bias_per_channel
    x = requantize(acc, shift_input + shift_weight, act_shifts["conv1"])
    x = relu(x)
    x = avgpool_int(x)
    shift_input = act_shifts["conv1"]

    # Conv2, Requantize, ReLU, AvgPool
    weight, shift_weight, bias, shift_bias = get("conv2")
    shift_acc = shift_input + shift_weight
    bias_per_channel = align_bias(bias, shift_bias, shift_acc).view(1, -1, 1, 1)
    acc = F.conv2d(x, weight) + bias_per_channel
    x = requantize(acc, shift_input + shift_weight, act_shifts["conv2"])
    x = relu(x)
    x = avgpool_int(x)
    shift_input = act_shifts["conv2"]

    # Output layer and requantization
    weight, shift_weight, bias, shift_bias = get("fc")
    x = x.flatten(1)
    acc = F.linear(x, weight) + align_bias(bias, shift_bias, shift_acc)
    logits = requantize(acc, shift_input + shift_weight, act_shifts["fc"])

    return logits


def c_array(f, ctype, name, tensor):
    # Layout = PyTorch memory order, e.g. weights [out_ch][in_ch][5][5]
    values = ", ".join(str(v) for v in tensor.flatten().to(torch.int32).tolist())
    f.write(f"static const {ctype} {name}[{tensor.numel()}] = {{{values}}};\n\n")

def c_header(f):
    f.write("// Generated by lenet1.py\n")
    f.write("#pragma once\n#include <stdint.h>\n\n")


def export_weights(weights, act_shifts, path):
    with open(path, "w") as f:
        c_header(f)
        # Activation shifts (from calibration)
        for point, n in act_shifts.items():
            f.write(f"#define SHIFT_{point.upper()} {n}\n")
        f.write("\n")
        # Weights and biases: model.conv1.weight -> CONV1_WEIGHT[] + CONV1_WEIGHT_SHIFT
        for name, t in weights.items():
            cname = name.replace("model.", "").replace(".", "_").upper()
            f.write(f"#define {cname}_SHIFT {t['shift']}\n")
            c_array(f, "int8_t", cname, t["q"])


def export_testdata(loader, weights, act_shifts, path, num_images=20):
    images, labels = next(iter(loader))
    images, labels = images[:num_images], labels[:num_images]
    with open(path, "w") as f:
        c_header(f)
        f.write(f"#define NUM_IMAGES {num_images}\n\n")
        # Already quantized: C starts from bit-identical inputs
        c_array(f, "int8_t", "TEST_IMAGES", quantize_tensor_int8(images, act_shifts["input"]))
        c_array(f, "uint8_t", "TEST_LABELS", labels)
        # Expected final outputs for all images
        c_array(f, "int8_t", "GOLDEN_LOGITS", forward_int8(images, weights, act_shifts))


def main():
    # Reproducibility: same weights, same val split, same shifts on every run
    torch.manual_seed(0)

    train_data = datasets.MNIST(
        root='./data',
        train=True,
        # Transform PIL images to tensors [1, 28, 28] with scaled values (pixel/255)
        transform=transforms.ToTensor(),
        download=True)
    test_data = datasets.MNIST(
        root='./data',
        train=False,
        # Transform PIL images to tensors [1, 28, 28] with scaled values (pixel/255)
        transform=transforms.ToTensor(),
        download=True)
    train_data, val_data = random_split(train_data, [1 - VAL_FRAC, VAL_FRAC])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=1000, shuffle=False,  num_workers=4)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False, num_workers=4)

    # Train lenet model and evaluate accuracy against test data
    model = LeNet1()
    # Skip training if a trained checkpoint already exists
    if os.path.exists("checkpoints/lenet1.pt"):
        print("Loading existing checkpoint lenet1.pt (delete to retrain)")
        model.load_state_dict(torch.load("checkpoints/lenet1.pt"))
    else:
        print("No checkpoint found, training from scratch")
        os.makedirs("checkpoints", exist_ok=True)
        train(model, train_loader, val_loader)
    accuracy = evaluate(model, test_loader)
    print(f"Float accuracy: {accuracy:.4f}")

    # Quantize weights, determine activation shifts and evaluate quantized models accuracy
    weights = quantize_model(model.state_dict())
    act_shifts = calibrate(model, val_loader)
    torch.save({"weights": weights, "act_shifts": act_shifts}, "checkpoints/lenet1_int8.pt")
    int8_accuracy = evaluate_int8(weights, act_shifts, test_loader)
    print(f"Int8 accuracy: {int8_accuracy:.4f}")

    # Export weights and test data
    os.makedirs("export", exist_ok=True)
    export_weights(weights, act_shifts, "export/weights.h")
    export_testdata(test_loader, weights, act_shifts, "export/testdata.h")
    print("Weights and testdata saved to /export.")


if __name__ == "__main__":
    main()