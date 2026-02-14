# gradient-boosting
Gradient Boosting Regressor from Scratch
This repository contains a Python implementation of the Gradient Boosting Regressor algorithm, built from the ground up using only numpy and pandas.

The goal of this project is to demystify the "black box" of ensemble learning by breaking down the additive modeling process step-by-step.

The Algorithm
This implementation follows the standard Forward Stagewise Additive Modeling approach. Unlike Random Forest, which builds trees in parallel, Gradient Boosting builds trees sequentially, where each new tree corrects the errors (residuals) of the previous ensemble.

