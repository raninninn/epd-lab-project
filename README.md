# Final project for the Customized Embedded Processor lab at CES
## What does this do?
Implement an accelerator for a LeNet1 type neural network on the RealDigital Blackboard, using both PS and PL. The network is quantized with integer quantization. Weights are stored in BRAM.
On a Ryzen 8500G, we can run the C implementation in 4.66 seconds.
## How can I reproduce the test environment?
1. Export all test inputs.
```
$ cd data_export && python mnist_exporter.py
```
2. Train reference network and export weights
```
$ cd ../weight_export && python network.py --train
```
3. Build executables for all inputs
```
$ cd ../lenet1 && ./build_script
```
## How do I verify it?
After setting up the test environment,
1. Get predictions from the Python reference implementation:
```
$ cd weight_export && python network.py --report-predictions > preds.txt
```
2. Compile with `-DPRINT_LABELS`
3. Get predictions from the C implementation
```
$ cd ../lenet1 && time ./a.out > preds.txt
```
4. Compare using diff
```
$ diff -q preds.txt ../weight_export/preds.txt
```
## Gotchas
- This implementation uses ReLU instead of tanh.
- This implementation uses banker's rounding for all divisions to match PyTorch's behaviour.
