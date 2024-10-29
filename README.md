# StableBehavior.jl

Code for the experiments in "[Stabilizing reinforcement learning control: A modular framework for optimizing over all stable behavior](https://doi.org/10.1016/j.automatica.2024.111642)".

The examples from the paper are [here](./examples) with supporting code under [src](./src).

![Alt text](/src/misc/anim1.gif)

## Installation
Clone this repo, navigate to the directory in the Julia REPL, then type the commands:
```
]
activate .
instantiate
```
Note I used Julia 1.8.

## Citation
Please use the following bib entry:
```
@article{lawrence2024stabilizing,
  title={Stabilizing reinforcement learning control: A modular framework for optimizing over all stable behavior},
  author={Lawrence, Nathan P and Loewen, Philip D and Wang, Shuyuan and Forbes, Michael G and Gopaluni, R Bhushan},
  journal={Automatica},
  year={2024},
  doi = {https://doi.org/10.1016/j.automatica.2024.111642}
}
```

---
### Note about [`stablePID.jl`](/src/policies/stablePID.jl)

- Equation 15 in the paper is a special case of Equation 4 in [Furieri *et al.*](http://arxiv.org/abs/1903.03828) for SISO systems.
- However, the code in `stablePID.jl` is written in general terms to match Equation 4. This leads to some redundancy but was done to show that the approach is not limited.
- There are two basic ideas in the code:
  1. Enforce that the control law $YX^{-1}$ works out to be a PI controller.
  2. Use Algorithm 1 in our paper to simulate and enforce that system responses behave like the dynamic relationship in Equation 15 (4 in Furieri *et al.*). This acts as a surrogate for the transfer function formulation.
