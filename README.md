# Learning Control-Oriented Dynamical Structure from Data
This repository accompanies the paper ["Learning Control-Oriented Dynamical Structure from Data"](https://arxiv.org/abs/2302.02529) [1].


## Getting started
Ensure you are using Python 3. Clone this repository and install the packages listed in `requirements.txt`. In particular, this code uses [JAX](https://github.com/google/jax), [Equinox](https://github.com/patrick-kidger/equinox), and [CasADi](https://web.casadi.org/).


## Reproducing results
Trajectory and data generation via MPC for both training and testing can be done with the command `./auto_datagen.sh`. This may take a few minutes.

Model training and testing (for various training set sizes and random seeds) can be done with the command `./auto_traintest.sh`. This will take a long time, so feel free to edit `./auto_traintest.sh` if you just want to try, e.g., a single training set size and seed.

Testing for the planar quadrotor on the double loop-the-loop trajectory can be done with the command `./auto_test_double_loop.sh`.

Finally, the figures in [1] can be reproduced and customized with the [Jupyter](https://jupyter.org/) notebook `plots.ipynb`.


## Citing this work
Please use the following BibTex entry to cite this work.
```
@INPROCEEDINGS{RichardsSlotineEtAl2023,
author      = {Richards, S. M. and Slotine, J.-J. and Azizan, N. and Pavone, M.},
title       = {Learning control-oriented dynamical structure from data},
year        = {2023},
booktitle   = {International Conference on Machine Learning (ICML)},
url         = {https://arxiv.org/abs/2302.02529},
doi         = {10.48550/arXiv.2302.02529},
}
```


## References
[1] S. M. Richards, J.-J. Slotine, N. Azizan, and M. Pavone. ["Learning control-oriented dynamical structure from data"](https://arxiv.org/abs/2302.02529). In *International Conference on Machine Learning (ICML)*, 2023.
