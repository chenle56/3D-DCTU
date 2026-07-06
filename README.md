# 3D-DCTU Source Code Package

This package provides the source code for training and testing the proposed 3D-DCTU network for 3D seismic suppression.

## 1. Package structure

```text
.
├── data/                  # Example data provided for training and testing
├── dcn/                   # Deformable convolution-related modules
├── deformable_LKA/        # Deformable large-kernel attention-related modules
├── func/                  # Utility functions
├── models/                # Network model definitions
├── FCNVMB_train.py        # Training script
├── FCNVMB_test_patch.py   # Testing script
├── LibConfig.py           # Library configuration
├── ParamConfig.py         # Parameter configuration
├── PathConfig.py          # Path configuration
├── requirements.txt       # Required Python packages
└── README.md              # Instruction file
```

## 2. Requirements

The code is implemented in Python. The required packages are listed in `requirements.txt`.

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

Please make sure that the Python environment and deep-learning framework are properly configured before running the training or testing scripts.

## 3. Data preparation

Example data are provided in the `data/` folder. These data can be used to run the training and testing workflow directly.

Users may also prepare their own data for training and testing. When using custom data, please organize the data following the same structure and format as the example data in the `data/` folder, and modify the corresponding paths and parameters in `PathConfig.py` and `ParamConfig.py` if necessary.

## 4. Training

To train the network, run:

```bash
python FCNVMB_train.py
```

The training parameters, input/output paths, and related settings can be modified in:

```text
ParamConfig.py
PathConfig.py
LibConfig.py
```

## 5. Testing

To test the trained network using patch-based prediction, run:

```bash
python FCNVMB_test_patch.py
```

The testing data path, model path, and output path can be modified in the configuration files.

## 6. Notes

This source-code package is provided to facilitate reproducibility and review of the proposed algorithm. The main training and testing workflows are implemented in `FCNVMB_train.py` and `FCNVMB_test_patch.py`, respectively. The provided example data in the `data/` folder can be used to verify the execution of the code. Users can also generate or prepare their own seismic data pairs and use them to train and test the network.
