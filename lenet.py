import os
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torchvision
import numpy as np
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
from brevitas.core.restrict_val import RestrictValueType
import argparse
import numpy as np
import time

# I use Power-of-Two Quantization for hardware efficiency (multiplications turn into shift operations)
# this limits the network weights to values whose magnitudes are exact powers of two
class Int8WeightPerTensorPoT(Int8WeightPerTensorFloat):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
class Int8ActPerTensorPoT(Int8ActPerTensorFloat):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO

def parity_transform(img):
    arr = (np.array(img, dtype=np.uint8) // 2).astype(np.int8)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)


# this function converts a PyTorch tensor into a flattened C array header file
def export_tensor_to_c_header(tensor, var_name, filename, c_type="int32_t"):
    # flatten tensor and convert to numpy array of integers
    flat_data = tensor.detach().cpu().int().numpy().flatten()
    shape_size = len(flat_data)
    
    # format as C array string
    array_str = np.array2string(flat_data, max_line_width=np.inf, separator=', ', threshold=np.inf)
    array_str = array_str.replace('[', '{').replace(']', '}')
    
    with open(filename, "w") as f:
        f.write("#include <stdint.h>\n\n")
        f.write(f"// Golden Reference Output for {var_name}\n")
        f.write(f"const {c_type} {var_name.upper()}[{shape_size}] = {array_str};\n")
    
    print(f"Exported {filename} ({shape_size} elements)")

# network definition
class LeNet(Module):
    def __init__(self, weight_bit_width=8, act_bit_width=8):

        super(LeNet, self).__init__()

        self.in_quant = qnn.QuantIdentity(act_quant=Int8ActPerTensorPoT, bit_width=act_bit_width, return_quant_tensor=True)
        self.out_quant = qnn.QuantIdentity(act_quant=Int8ActPerTensorPoT, bit_width=act_bit_width, return_quant_tensor=True)

        self.conv1 = qnn.QuantConv2d(in_channels=1, out_channels=4, bias=False, kernel_size=(5,5), weight_bit_width=weight_bit_width, weight_quant=Int8WeightPerTensorPoT)
        self.conv2 = qnn.QuantConv2d(in_channels=4, out_channels=12, bias=False, kernel_size=(5,5), weight_bit_width=weight_bit_width, weight_quant=Int8WeightPerTensorPoT)

        self.pool = qnn.TruncAvgPool2d(2)  # self.pool = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.fc0 = qnn.QuantLinear(in_features=4*4*12, out_features=10, bias=False, weight_bit_width=weight_bit_width, weight_quant=Int8WeightPerTensorPoT)
        
        self.relu1 = qnn.QuantReLU(act_quant=Int8ActPerTensorPoT, bit_width=act_bit_width, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(act_quant=Int8ActPerTensorPoT, bit_width=act_bit_width, return_quant_tensor=True)
    
    def forward(self,x):
        x = x.view(x.shape[0], 1, 28, 28)

        x = self.in_quant(x)        

        x = self.conv1(x)
        x = self.relu1(x)           # (B,24,24,4)

        x = self.pool(x)            # (B,12,12,4)

        x = self.conv2(x)
        x = self.relu2(x)           # (B,8,8,12)

        x = self.pool(x)            # (B,4,4,12)

        x = x.view(x.shape[0], -1)  # flatten -> 192

        x = self.fc0(x)

        x = self.out_quant(x)
        
        return x


def numpy_to_c_array(arr, var_name="my_array", c_type="int8_t"):
    shape = 1
    for d in arr.shape:
        shape *= d
    body = np.array2string(arr, max_line_width=np.inf, separator=', ', threshold=np.inf)
    body = body.replace('[', '{').replace(']', '}')
    return f"const {c_type} {var_name.upper()}[{shape}] = {body};\n"

def export_weights(model_layer, layer_name, output_dir="weights_export"):
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"{layer_name}.h")
    tensor = model_layer.quant_weight()
    
    with open(fname, "w") as f:
        f.write("#include <stdint.h>\n\n")
        f.write(numpy_to_c_array(tensor.int().cpu().numpy().flatten(), layer_name))
        f.write(f"const int8_t {layer_name.upper()}_SCALE = {(-torch.log2(tensor.scale)).int().item()};\n")


if __name__ == "__main__":
    ############ Setup and Preprocessing ############
    parser = argparse.ArgumentParser(description="Quantized model for keyword spotting")
    parser.add_argument("filename", type=str, nargs='?', help="Output file name")
    parser.add_argument("--epochs", type=int, nargs='?',default=50, help="Number of epochs")
    parser.add_argument("--gpu", type=str, nargs='?',default=0, help="Index of GPU if available")
    parser.add_argument("--train", action='store_true', help="Train model")
    parser.add_argument("--seed", type=int, nargs='?',default=5, help="Random seed")
    parser.add_argument("--lr", type=float, nargs="?", default=0.001, help="Learning rate")
    parser.add_argument("--bsize", type=int, nargs='?', default=256, help="Batchsize")
    parser.add_argument("--wbits", type=int, default=8, help="Weight bit width")
    parser.add_argument("--abits", type=int, default=8, help="Activation bit width")

    args = parser.parse_args()
    bsize = args.bsize
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Device and Model Declaration
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    print(device)
    model = LeNet(weight_bit_width=args.wbits, act_bit_width=args.abits).to(device)


    # Load and split training data
    train_set   = torchvision.datasets.MNIST(root='./data',train=True, download=True, transform=parity_transform)   # train=True loads the 60,000 training samples
    test_set    = torchvision.datasets.MNIST(root='./data',train=False, download=True, transform=parity_transform)  # train=False loads the 10,000 test samples

    # encapsulate data into dataloader form
    num_workers = 1
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bsize, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=bsize, shuffle=False, num_workers=num_workers)
    
    # Loss Function, Optimizer, Scheduler
    error = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("models", exist_ok=True)
    save_path = f"models/{args.filename}.pt"

    ############ Training ############

    if args.train:
        highestAcc = 0.0

        start_time = time.time()

        for epoch in range(args.epochs):
            model.train()
            train_correct = 0
            train_total = 0
            for images, labels in train_loader:
                # Transfering images and labels to GPU if available
                images, labels = images.to(device), labels.to(device)
            
                prediction = model.forward(images)

                loss = error(prediction.value, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    pred = prediction.value.argmax(1, keepdim=True)
                    train_correct += pred.eq(labels.data.view_as(pred)).sum().item()
                    train_total += images.size(0)

            train_acc = (train_correct * 100.0) / train_total
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                 for images, labels in test_loader:
                     images, labels = images.to(device), labels.to(device)
                     prediction = model.forward(images)
                     pred = prediction.value.argmax(1, keepdim=True)
                     val_correct += pred.eq(labels.data.view_as(pred)).sum().item()
                     val_total += images.size(0)
            
            val_acc = (val_correct * 100.0) / val_total
            print(f"Epoch: {epoch+1}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            if val_acc > highestAcc:
                highestAcc = val_acc
                torch.save(model.state_dict(), save_path)
                print(f"--> Saved new best model to {save_path}")

        elapsed_time = time.time() - start_time
        print("-" * 50)
        print(f"Total Training Time: {elapsed_time/60:.2f} minutes")

        # I think it is not needed here, but for completeness, consider doing extending with a test set check later


    ############ Parameter Export ############
    print("\n--- Exporting Weights to C Headers ---")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    
    # Export Layer Weights
    export_weights(model.conv1, "weights_c1")
    export_weights(model.conv2, "weights_c3")
    export_weights(model.fc0, "weights_out")
    
    # extract Activation Scales by pushing a dummy batch to calculate scales
    with torch.no_grad():
        dummy_input = torch.randint(0, 128, (1, 1, 28, 28), dtype=torch.float32).to(device)
        _ = model(dummy_input) # Trigger quantization tracking
        
        s0 = model.in_quant.act_quant.scale()
        s1 = model.relu1.act_quant.scale()
        s2 = model.relu2.act_quant.scale()
        s3 = model.out_quant.act_quant.scale()
        
        act_scales_path = "weights_export/act_scales.h"
        with open(act_scales_path, "w") as f:
            f.write("#include <stdint.h>\n\n")
            f.write(f"#define INPUT_ACT_SHIFT {(-torch.log2(s0)).int().item()}\n")
            f.write(f"#define FIRST_ACT_SHIFT {(-torch.log2(s1)).int().item()}\n")
            f.write(f"#define SECOND_ACT_SHIFT {(-torch.log2(s2)).int().item()}\n")
            f.write(f"#define OUTPUT_ACT_SHIFT {(-torch.log2(s3)).int().item()}\n")

        
    print("Export complete! Files are saved in the 'weights_export' directory.")

    ############ Golden Debug Headers Generation ############
    print("\n--- Generating Golden Debug Headers ---")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        # get first image from test set
        test_image, test_label = test_set[0]
        input_tensor = test_image.unsqueeze(0).to(device)

        # we need to match exact parity transform for C input vector
        raw_input_data = test_image.squeeze().cpu().numpy().astype(np.int8).flatten()
        export_tensor_to_c_header(torch.tensor(raw_input_data), "test_input_0", "weights_export/test_input_0.h", c_type="int8_t")
        
        with open("weights_export/test_label_0.h", "w") as f:
            f.write(f"#define EXPECTED_LABEL {test_label}\n")

        # we do a step-by-step forward pass and save header files at each step/layer
        x = model.in_quant(input_tensor)
        
        x = model.conv1(x)
        x = model.relu1(x)
        export_tensor_to_c_header(x.int(), "c1_out", "weights_export/c1_out.h")
        
        x = model.pool(x)
        export_tensor_to_c_header(x.int(), "s2_out", "weights_export/s2_out.h")
        
        x = model.conv2(x)
        x = model.relu2(x)
        export_tensor_to_c_header(x.int(), "c3_out", "weights_export/c3_out.h")
        
        x = model.pool(x)
        export_tensor_to_c_header(x.int(), "s4_out", "weights_export/s4_out.h")
        
        x = x.view(x.shape[0], -1)
        x = model.fc0(x)
        x = model.out_quant(x)
        export_tensor_to_c_header(x.int(), "fc_out", "weights_export/fc_out.h")


    print("\n--- Exporting 10 Test Images for Milestone 2 ---")
    NUM_IMAGES = 10

    with open("weights_export/test_batch.h", "w") as f:
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define NUM_TEST_IMAGES {NUM_IMAGES}\n\n")
        
        # write ground truth labels array
        labels = [test_set[i][1] for i in range(NUM_IMAGES)]
        labels_str = "{" + ", ".join(str(l) for l in labels) + "}"
        f.write(f"const int8_t TEST_LABELS[{NUM_IMAGES}] = {labels_str};\n\n")
        
        # write flat array of all 10 images (10 * 784 = 7840 elements)
        f.write(f"const int8_t TEST_IMAGES[{NUM_IMAGES}][784] = {{\n")
        for i in range(NUM_IMAGES):
            img_tensor, _ = test_set[i]
            # musst match parity transform integer domain
            raw_data = img_tensor.squeeze().cpu().numpy().astype(np.int8).flatten()
            f.write("  {" + ", ".join(map(str, raw_data)) + "}" + ("," if i < NUM_IMAGES - 1 else "") + "\n")
        f.write("};\n")

    print("Exported weights_export/test_batch.h successfully!")


    print("\nAll weights, activation shifts, and golden debug headers are now exported!")