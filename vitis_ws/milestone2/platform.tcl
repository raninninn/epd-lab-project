# 
# Usage: To re-create this platform project launch xsct with below options.
# xsct /home/ran/Documents/uni/s10/asip_lab/vitis_ws/milestone2/platform.tcl
# 
# OR launch xsct and run below command.
# source /home/ran/Documents/uni/s10/asip_lab/vitis_ws/milestone2/platform.tcl
# 
# To create the platform in a different location, modify the -out option of "platform create" command.
# -out option specifies the output directory of the platform project.

platform create -name {milestone2}\
-hw {/home/ran/Documents/uni/s10/asip_lab/final_project/design_1_wrapper.xsa}\
-proc {ps7_cortexa9_0} -os {standalone} -out {/home/ran/Documents/uni/s10/asip_lab/vitis_ws}

platform write
platform generate -domains 
platform active {milestone2}
domain active {zynq_fsbl}
bsp reload
platform generate
platform active {milestone2}
platform config -updatehw {/home/ran/Documents/uni/s10/asip_lab/final_project/design_1_wrapper.xsa}
platform generate -domains 
platform active {milestone2}
platform config -updatehw {/home/ran/Documents/uni/s10/asip_lab/final_project/design_1_wrapper.xsa}
platform generate -domains 
platform active {milestone2}
platform generate -domains 
platform active {milestone2}
domain active {standalone_domain}
bsp reload
domain active {zynq_fsbl}
bsp reload
domain active {standalone_domain}
bsp config stdin "ps7_uart_0"
bsp config stdout "ps7_uart_0"
bsp config stdin "ps7_coresight_comp_0"
bsp write
bsp reload
catch {bsp regenerate}
platform generate -domains standalone_domain 
bsp config stdout "ps7_coresight_comp_0"
bsp write
bsp reload
catch {bsp regenerate}
platform generate -domains standalone_domain 
bsp reload
bsp config stdin "ps7_uart_0"
bsp config stdout "ps7_uart_0"
bsp write
bsp reload
catch {bsp regenerate}
platform generate -domains standalone_domain 
bsp config stdin "ps7_coresight_comp_0"
bsp config stdout "ps7_coresight_comp_0"
bsp write
bsp reload
catch {bsp regenerate}
platform generate -domains standalone_domain 
platform -make-local
platform generate -domains 
