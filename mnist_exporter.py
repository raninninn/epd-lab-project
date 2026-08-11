import numpy as np
import argparse
from pathlib import Path
from torchvision.datasets import MNIST

argparser = argparse.ArgumentParser(description="Export the test partition of the MNIST dataset as a C header file. Inputs are given as INPUT_TEST_X, labels are given as LABEL_TEST_X.")
argparser.add_argument("-o", "--output", type=str, default=Path("inputs"), help="Output path to the folder for the data (default: inputs)")
argparser.add_argument("-t", "--data-type", type=str, default="param_t", help="Quantization data type (default: i8)")
args = argparser.parse_args()

# transform images to flattened numpy arrays
def to_np_array(img):
    return (np.array(img, dtype=np.uint8) // 2).astype(np.int8).flatten()

train_dataset = MNIST(root="./data", train=True, download=True, transform=to_np_array)
test_dataset = MNIST(root="./data", train=False, transform=to_np_array)

args.output.mkdir(parents=True, exist_ok=True)

def export_dataset(dataset, name):
    for i, v in enumerate(dataset):
        img = v[0]
        label = v[1]
        filename = f"{name}_{i}.h"
        with open(args.output / filename, "w") as f:
            f.write("#include <stdint.h>\n\n")
            f.write(f"#define LABEL_{name.upper()}_{i} {label}\n")
            f.write(f"{args.data_type} INPUT_{name.upper()}_{i}[{img.shape[0]}] = {{")
            for j in range(img.shape[0]):
                f.write(str(img[j]))
                if j < img.shape[0] - 1:
                    f.write(", ")
            f.write("};\n")

export_dataset(test_dataset, "test")
#export_dataset(train_dataset, "train")
