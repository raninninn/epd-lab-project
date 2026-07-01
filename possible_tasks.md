# Milestone 1

I would like it if we can make the implementations parameterizable for extensibility and maintainability. Otherwise, we would need to agree on the fixed point scales to use beforehand.

## Writing a LeNet for milestone 1
- 1 afternoon at most
- could use external libraries like Eigen:
  - would that be reusable on a baremetal ARM platform like the Blackboard?
- Matthias already has some header files that contain the inputs and labels of MNIST, can be reused
@ali: The project slides mention a "pure C" implementation. Eigen is a C++ template library, so I am not sure we can use it. We will have to ask the TA.

## Procure comparison network
- 1 afternoon
- python implementation
- needs to use same weights as weights procured in task "Procure quantized weights"
- need to dump intermediate tensor outputs after every layer to text files for easier debugging if needed

## Procure quantized weights
- 1 afternoon at most, probably less
- pytorch has quantization stuff built in
- export as header files?
- dependency or subtask of task "Procure comparison network"

## Verify and potentially fix C impl
- 1 longer afternoon
- trivial subtask: measure runtime on x86

Proposed Task Division: Member 1 works on the python implementation, Member 2 works on C implementation
Member 3 can integrate both, but I would discourage that and instead let Member 1 or 2 or both of them do the integration.

# Milestone 2
## Write BRAM storage for weights
- 1 short afternoon?
- tightly coupled to task "Integrate BRAM weight storage"
- There's a [BRAM controller IP](https://docs.amd.com/v/u/en-US/pg078-axi-bram-ctrl)
- There's also the [Block Memory Generator IP](https://docs.amd.com/v/u/en-US/pg058-blk-mem-gen), which goes with the BRAM controller

## Integrate BRAM weight storage
- 1 afternoon
- Probably makes more sense if turned into subtask of task "Write BRAM storage for weights"

## Get runtime on ARM
- 1 h max

# Milestone 3
## Identify acceleration potential
- 10 minutes
- Probably just convolution anyway

## Write accelerators
- 1 afternoon, maybe 2 per accelerator
@ali: I would say that is a bit optimistic, especially if someone would like to give it a try without LLM assisstance
- needs to be integrated with BRAM-based weight storage
- might be bundled with weight storage? I.e., if we accelerate all our convolutions, there's no point in not bundling the convolution accelerator with its own BRAM-based weight storage

## Integrate accelerator in C code
- 1 afternoon
- obviously closely connected to writing the accelerator

## Get runtime on combined PS and PL
- 1 h max

# Presentation
## Prepare individual slides
- 1--2 afternoons? depends on how each individual likes to work on slides I guess
## Discuss slides
- 1--2 h
## Join slides and make it pretty
- 1 afternoon
