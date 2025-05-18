Here is the supplementary program and dataset for the manuscript "Modeling Nonstationary Extremal Dependence via Deep Spatial Deformations".


Folders and documents in Scripts&Codes:
1. AppData: application data and modelling results
2. SimData: simulated data and modelling results
3. Functions: functions for modelling that will be introduced in application_UKpr.R and simulation_Archi3.R
4. application_UKpr.R: the script for the data application with UK precipitation data. The required data must be downloaded from [Zenodo](https://zenodo.org/records/15459157) and placed in AppData/. 
5. simulation_Archi3.R: the script for a simulation demo, which can simulate r-Pareto processes with the provided risk functional and other hyperparameters. As the simulation of r-Pareto processes using max-functional is time-consuming, we also provide simulated data used in our simulation study, which must be downloaded from [Zenodo](https://zenodo.org/records/15459157) for site-, max-, and sum-functionals, and placed in SimData/. 


Remarks: Current program works for Tensorflow version 2.11.0 and Python version 3.7.11. Some updated Tensorflow version, e.g., 2.19.0, is not compatible. Implementing scripts requires the installation of the Python language, Tensorflow, and TensorFlow packages in R.


How to Start:
1. Install the corresponding software and packages.
2. Implement simulation_Archi3.R or application_UKpr.R for the simulation demo or the data application.
