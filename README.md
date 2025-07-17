Most of the updates are under the TIM folder.

Work flow:
Simulate data and train for mcmc results from bayesian.py. Data is stored in folders created with timestamped names.
After getting the mcmc results, run MIQP optimization from optimization.py. Need to change the argument of mcmc results dir.
