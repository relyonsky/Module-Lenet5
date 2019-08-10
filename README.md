The code can run in Vivado HLS

We realize the Lenet-5 on Zynq 7035. The floder Lenet is the fully Lenet-5 Network. It consists of two conv layers, two pooling layers & three fullyconnect layers. We realize the network through module-level pipeline. We divide Lenet-5 to three parts according to FLOPs. Each part can run in a FPGA, increasing hardware resources for parallel expansion

Each part has their own codes in floders, and codes need to run in Vivado HLS
