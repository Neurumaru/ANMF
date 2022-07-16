# ANMF
This is our implementation for the paper "Additional Neural Matrix Factorization model for Computational drug repositioning. BMC Bioinformatics"   
- Paper: Additional Neural Matrix Factorization model for Computational drug repositioning. BMC Bioinformatics
- Author: Xin-Xing Yang Southeast University    
- Email: 220174323@seu.edu.cn   

# Environment Settings
We use Keras with Theano as the backend. 
- Keras version:  '1.0.7'
- Theano version: '0.8.0'

# Example to run the codes.
Open the "ANMF.py" file and run it directly. In the Data folder is the test example we provide.

# Dataset
We provide two processed datasets: Gottlieb dataset and Cdatasets. 

# Modified by Neurumaru
We make this implementation work even on large-scale data
- Optimize DataLoader.py (original: Dataset.py)
- Use Tensorflow Dataset (new Dataset.py)
- Separate predict (evaluate.py) and calculate AUC (ANMF-AUC.py)
