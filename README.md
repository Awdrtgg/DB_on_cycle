# DB_on_cycle

This repo contains the source code that realizes the simulation and the algorithm proposed in the paper "Close spatial arrangement of mutants favors and disfavors fixation" by Yunming Xiao and Bin Wu. The paper will appear on PLOS Computational Biology.

#### fp_and_derivative.py
The realization of the proposed algorithm to calculate the fixation probabilities and their derivatives. Note that the sparse matrix method is not adopted here. You may ask for the sparse matrix version by emailing at the authors.

#### fixation_time.py
The code of theh proposed algorithm for calculating the conditional fixation time. 

#### simulation.py
The simulation code to evaluate the effectiveness of the algorithm.

#### simu_error_bar.py
To obtain the numerical statistics (like mean and variance) of the simulation.
