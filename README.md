
**Code is being refactored now.**

# Project Description

## Project Structure

```
Project_root
  |-Data_Generator                  // Encoding software for prediction data generation
  |   |-config.txt                  // Configuration file (can specify location during execution)
  |-Fitting_model                   // Stored model for grid search + cross-validation
  |-Raw_Mole_Encode_Generator       // Converts molecular structure files (SMILES, CML, MOL) to molecular encoding
  |-Data                            // Raw data storage location
  |-README.md                       // This file
```

## Operating System and Configuration

Linux (Recommend: Debian12), 16G RAM + 256G swap, CUDA11.8

## Language Version and Major Package Dependencies

### C++17

- CMake 3.25
- Clang 14.0.6
- GCC 11.3

### Python3.9

- PyTorch 2.2.2
- torcheval 0.0.7
- NumPy 1.26.4
- Scikit-learn 1.4.1
- Pandas 2.2.1
- Matplotlib 3.7.3
- shap 0.42.1

## Usage

1. Data preprocessing + one-hot encoding  
   `Data_Generator/build/Onehot_Generator` (compile to build folder)

2. Feature filtering, model selection, ensemble model training  
   `Fitting_model/essemble_model.py`

3. Data generation and prediction  
   Data generation:  
   `Data_Generator/build/Predictor`  
   Data prediction:  
   `Fitting_model/essemble_model.py`
