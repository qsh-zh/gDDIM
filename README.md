# gDDIM: Generalized denoising diffusion implicit models

[Qinsheng Zhang](https://qsh-zh.github.io/), [Molei Tao](https://mtao8.math.gatech.edu/), [Yongxin Chen](https://yongxin.ae.gatech.edu/)

We unbox the accelerating secret of DDIMs based on Dirac approximation and generalize it to general diffusion models. When
applied to the critically-damped Langevin diffusion model, it achieves an FID score of 2.26 on CIFAR10 with 50 steps.

![gDDIM](assets/fig1.png) 
![dirac](assets/fig2.png)

# Setup

## Docker

## From scratch

# Reproduce results

## CLD

Download the [checkpoint]() and evaluate FID
> the checkpoint has 2.2565 FID in my machine with 50 NFE

```shell
# todo
```

# Reference

```tex
@misc{zhang2022gddim,
      title={gDDIM: Generalized denoising diffusion implicit models}, 
      author={Qinsheng Zhang and Molei Tao and Yongxin Chen},
      year={2022},
      eprint={2206.05564},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```tex
@inproceedings{dockhorn2022score,
    title={Score-Based Generative Modeling with Critically-Damped Langevin Diffusion},
    author={Tim Dockhorn and Arash Vahdat and Karsten Kreis},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2022}
}
```
