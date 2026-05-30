========================================================================
    PLAID NILM (CLS_13k_NPU) PyTorch Training & Evaluation Project
========================================================================

This repository provides the PyTorch implementation, training pipeline, and 
evaluation tools for the "Non-Intrusive Load Monitoring (NILM) with the PLAID 
Dataset: Time Series Classification for Appliance Detection" model. 

It converts the original TinyML-based model from Texas Instruments (TI) TensorLab 
(the CLS_13k_NPU model with 13K parameters) into a standard PyTorch implementation, 
trained and evaluated on the plaid_nilm_submetered_dataset.

Reference to the original TI implementation:
https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/PLAID_nilm_classification

------------------------------------------------------------------------
1. Repository Directory Structure
------------------------------------------------------------------------
* train.py                 : Main script for model definition, dataset downloading, 
                             feature extraction, training, and testing.
* best_model.pth           : PyTorch model weights achieving the best validation 
                             voted accuracy (98.49% on the full validation set).
* requirements.txt         : Lightweight Python package dependencies (NumPy, Pandas, PyTorch).
* readme.txt               : This instruction document (in English).
* plaid_dataset/annotations/: Pre-defined 8:2 split list annotations.
  ├── training_list.txt    : Training set file path list.
  └── validation_list.txt  : Validation (test) set file path list.

------------------------------------------------------------------------
2. Environment Setup & Installation
------------------------------------------------------------------------
This project is lightweight and does NOT require heavy frameworks like TensorFlow.
It only depends on standard scientific computing and PyTorch libraries.

Please run the following commands in your terminal (PowerShell or CMD):

1. Create and activate a Python virtual environment (Recommended):
   # Create virtual environment named nilm_env
   python -m venv nilm_env
   # Activate (PowerShell)
   .\nilm_env\Scripts\Activate.ps1
   # Activate (CMD)
   .\nilm_env\Scripts\activate.bat

2. Install dependencies:
   pip install -r requirements.txt

------------------------------------------------------------------------
3. Usage Instructions
------------------------------------------------------------------------
Ensure your virtual environment is active, then execute in the root directory:

1. Run Test and Evaluation (using the pre-trained best_model.pth):
   python train.py --test
   * This command will automatically download and extract the PLAID dataset 
     zip archive (~24MB) if it is missing, run inference on the validation 
     set, and print detailed global accuracy and class-wise metrics.

2. Start a New Training Process:
   - If you want to train from scratch, first delete the existing `best_model.pth`.
   - Then run:
     python train.py --train
   * The script will train for 30 epochs and automatically save the weights 
     with the highest validation accuracy as `best_model.pth`.

------------------------------------------------------------------------
4. Benchmark Results (Accuracy & Consistency)
------------------------------------------------------------------------
The CLS_13k_NPU model was evaluated with Float32 PyTorch. The benchmark results on 
a class-balanced subset of the validation dataset (30 files, 1290 frame-level samples)
are as follows:

A. CROSS-MODEL ACCURACY SUMMARY (File-Level Voting)
========================================================================
Model Version                       | Voted Accuracy (%)
------------------------------------------------------------------------
PyTorch Base Model (float32)        | 100.00%
========================================================================