# Learning Control-Oriented Dynamical Structure from Data

This repository accompanies the paper ["Learning Control-Oriented Dynamical Structure from Data"](https://arxiv.org/abs/2302.02529) [1].


## Getting started

Ensure you are using Python 3. Clone this repository and install the packages listed in `requirements.txt`. In particular, this code uses [JAX](https://github.com/google/jax), [Equinox](https://github.com/patrick-kidger/equinox), and [CasADi](https://web.casadi.org/).


## Reproducing results

Test trajectory generation via MPC can be done with the command `./auto_generate.sh`. This may take a few minutes.

Model learning and testing (for various training set sizes and random seeds) can be done with the command `./auto_traintest.sh`. This will take a long time, so feel free to edit `./auto_traintest.sh` if you just want to try, e.g., a single training set size and seed.

Testing for the planar quadrotor on the double loop-the-loop trajectory can be done with the command `./auto_test_double_loop.sh`.

Finally, the figures in [1] can be reproduced with the command `python plots.py`.


## Citing this work

Please use the following bibtex entry to cite this work.
```
@UNPUBLISHED{RichardsSlotineEtAl2023,
author    = {Richards, S. M. and Slotine, J.-J. and Azizan, N. and Pavone, M.},
title     = {Learning control-oriented dynamical structure from data},
year      = {2023},
note      = {Submitted},
url       = {https://arxiv.org/abs/2302.02529},
}
```


## References
[1] S. M. Richards, J.-J. Slotine, N. Azizan, and M. Pavone. Learning control-oriented dynamical structure from data. 2023. Submitted. Available at <https://arxiv.org/abs/2302.02529>.
