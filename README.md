#  Δ-Machine Learning of Triplet Excitation Energies in Organic Chromophores  
# Abstract
Here, we report Δ-machine learning approch for predicting triplet excitation energies (T1) of diverse organic chromohores by combining high quality reference data with quantum chemical data. A directed message passing neural network corrects TDDFT, ΔSCF and xTB/sTDA prediction to achieve near chemical accuracy while substantially reducing computational cost. Notably, Δ-ML corrected xTB/sTDA predictions close to TD-DFT quality, enabling rapid high-throughput screening of T1 energies.
# Dataset
The training and testing datasets used in this study for TDDFT, ΔSCF, and xTB/sTDA calculations are provided in the Dataset folder.
# Models
The pre-trained models for TDDFT, ΔSCF, and xTB/sTDA methods used in this study for predicting the T₁ energy are provided in the Models folder.
# Scripts
The automated Python scripts used to perform TDDFT and ΔSCF calculations with Gaussian 16, along with the automated Python code for calculating T₁ energies using the xTB/sTDA method, are available in the Scripts folder.
# Code
The code used for prediction with the pre-trained models is provided in the "prediction_code.ipynb" Jupyter notebook.
