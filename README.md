# EBI_DLemulator
Deep Residual Regression Neural Network from the paper “Emulation of an atmospheric gas-phase chemistry solver through deep learning: Case study of Chinese Mainland” in Keras is built for replacing and accelerating the gas-phase chemistry solution procedure in the chemical transport models.  
 
## Data
A state-of-the-art open-source CTM version of the Community Multiscale Air Quality (CMAQ) model (https://github.com/USEPA/CMAQ/tree/5.2) is selected to derive the training and validation data for the emulator of the gas chemistry solver.  

<img src="https://github.com/hazenet-cn/EBI_DLemulator/blob/main/images/fig%201.png"  width = "70%" height = "70%"/>

## Requirements
+ Tensorflow  
+ Keras  
+ Numpy  

## Usage
Replace 'BLD_CCTM_v521_intel_dl' with 'BLD_CCTM_v521_intel' in CMAQ, you may modify the path and a little bit of code as needed.
