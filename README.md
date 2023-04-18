# ADMM_PINNs
This repository contains the source code for the paper "The ADMM-PINNs Algorithmic Framework for Nonsmooth PDE-Constrained Optimization: A Deep Learning Approach" by Yongcun Song, Xiaoming Yuan, and Hangrui Yue. The paper can be found on [arXiv:2302.08309](https://arxiv.org/abs/2302.08309).
## Requirements
To run the code in this repository, you will need the following packages:

1.Jupyter Notebook

2.PyTorch

3.NumPy

4.SciPy

5.Plotly

6.Matplotlib

7.CUDA (Optional)

We recommend using Conda to manage your packages.
## Demo
Each '.ipynb' file in this repository can be executed in Jupyter Notebook, and its name corresponds to its functionality. There are also several '.py' files in the repository that serve as the foundation for implementing 2-dimensional finite element methods. Additionally, there are 'modelxx_norm.pth' files that contain pre-trained denoisers of CNNs. These files can be generated using 'generate_denoiser.ipynb', but you will need to download the BSD68 training set yourself. 'generate_denoiser.ipynb' is the PyTorch version of the original code found at https://github.com/cszn/DnCNN.

The four cases included are:
### 'Inverse_Potential'
The source code for solving the Inverse Potential Problem.
### 'Burgers'
The source code for solving the Control Constrained Optimal Control of the Burgers Equation.
### 'SourceID'
The source code for Discontinuous Source Identification for Elliptic PDEs.
### 'L1Control'
The source code for solving Sparse Optimal Control of Parabolic Equations. 
