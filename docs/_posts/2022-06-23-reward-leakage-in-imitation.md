# No rewards or expert? No problem.

This work is in progress, and highly experimental.

### Abstract

At the heart of Reinforcement Learning, is learning from evaluative feedback. This is typically specified in terms of a reward or cost function. In the absence of a reward function, an agent can learn from an expert (policy or demonstrations), either in a supervised manner (Behavioral Cloning), by first recovering a reward function that is optimal for the demonstrations (Inverse RL), or by learning to generate expert-like trajectories (Adversarial Imitation Learning).
We find that in some specific settings on sparse reward tasks, it is possible to train an agent solely by hallucinating a specific form of evaluative feedback or teacher. We discover this specific form, through a modification to the typical GAIL loss, using a somewhat pathological network. We find that this new loss leads to improved performance in grid world and 3D environment goal based tasks, using visual observations.

### Background

<p align="center" style="font-size:8px;">
<img src="https://github.com/amy12xx/amy12xx.github.io/blob/master/gail_loss_images/gail.png" width=360></img>
</br>
</p>

The commonly used GAIL (Discriminator) loss, is similar to the GAN Discriminator loss, optimizing a dual objective; i.e. increasing the cost of non-expert demonstrations, while decreasing the cost of expert demonstrations.

<p align="center" style="font-size:8px;">
<img src="https://github.com/amy12xx/amy12xx.github.io/blob/master/gail_loss_images/gail_loss.png" width=460></img>
</br>
<img src="https://github.com/amy12xx/amy12xx.github.io/blob/master/gail_loss_images/gail_reward.png" width=360></img>
</br>
</p>

In a similar manner, GAIL from observations, or GAILfO, has a similar objective, using the state occupancy measure, instead of the state-action occupancy measure used by GAIL. The generator in both cases, GAIL and GAILfO, uses the Discriminator as the reward or cost function.

<p align="center" style="font-size:8px;">
<img src="https://github.com/amy12xx/amy12xx.github.io/blob/master/gail_loss_images/gailfo_loss.png" width=360>
</br>
<img src="https://github.com/amy12xx/amy12xx.github.io/blob/master/gail_loss_images/gailfo_reward.png" width=160></img>
</br>
</p>

### Modifying GAILfO loss to hallucinate a teacher

We simplify the GAILfO loss by removing the loss component that optimizes the cost of expert demonstrations, so that the Discriminator can focus on the cost of a non-expert. In this case, the expert data is no longer used for training, and the problem ceases to be an occupancy matching problem, shown by Ho and Ermon to be the dual of the Inverse RL problem.

<p align="center" style="font-size:8px;">
<img src="https://github.com/amy12xx/amy12xx.github.io/blob/master/gail_loss_images/gailfo_reward2.png" width=160></img>
</br>
</p>

However this is observed to work, in specific settings, consistently, over hyper parameter sweeps, and in training. We first look at the results, and then discuss the settings in which it works.

## Experiment Results

### On sparse reward tasks

We conduct experiments using GAILfO as the baseline, and our modified loss. To test the sparse reward setting, we use the Minigrid (minimalistic gridworlds), and Miniworld (minimalistic 3D environment similar to DM Lab) tasks. We use pixel observations, with no action information. We use stacked observations for Miniworld, since it is a more difficult environment than gridworld. We train over three seeds.

In order to test how the policies generalize, we introduce randomness in the color attributes of the various objects in the environments. For e.g., in the 3D maze environments, the goal (box) is rendered with random colors at test time. During training as well as expert data generation, this is set to the default (red) color. In a similar manner, in the gridworld environments, various entity (door, key, wall) colors are randomized at test time. We use our best performing model to test for generalization, where possible.


<p align="center" style="font-size:8px;">
<img src="https://github.com/amy12xx/amy12xx.github.io/blob/master/gail_loss_images/minigrid and miniworld gail loss.jpg"></img>
</br>
Figure 1. Results on Minigrid, and Miniworld environments, using pixel observations, over 3 seeds.
</p>

<p align="center" style="font-size:8px;">
<img src="https://github.com/amy12xx/amy12xx.github.io/blob/master/gail_loss_images/wandb minigrid miniworld.jpg"></img>
</br>
Figure 2. W&B hyperparameter sweep results on Minigrid, and Miniworld environments, using pixel observations.
</p>

We run an equal number of sweeps for both methods, on all environments. On several environments, GAILfO fails to find parameters to learn a successful policy, whereas the new loss only fails on one environment (TMazeRight).

<p align="center" style="font-size:8px;">
<img src="https://github.com/amy12xx/amy12xx.github.io/blob/master/gail_loss_images/test performance.png"></img>
</br>
Figure 3. Evaluation on 100 episodes, using the best performing model, on (different) test seed, on the training environment (left), and generalized environment (right).
</p>


### On dense reward tasks

For the dense reward settings, we test using (state-action) demonstrations, on classic control tasks (CartPole, Acrobot and Pendulum), and using stacked pixel observations, on Mujoco tasks (Walker and Hopper).

<p align="center" style="font-size:8px;">
<img src="https://github.com/amy12xx/amy12xx.github.io/blob/master/gail_loss_images/classic.jpg"></img>
</br>
Results on dense reward tasks: Classic Control tasks (CartPole, Acrobot and Pendulum) using demonstrations (LfD) and Mujoco using pixel observations, over 3 seeds.
</p>

### Model and expert details

For the pixel observations, we use a network with two CNNs, with 32 and 64 kernels, and ReLU activations. On the state based demonstrations, we use a simpler multilayer perceptron. We run hyperparameter sweeps on both models, for all environments.

On all experiments, we use experts to generate optimal trajectories. Check the [Expert data](gail_loss/data/README.md) for more details.

## Analysis and Discussion



## Installation

- [Installation](docs/installation.md)

## Citation

If you use this repo in your research, please consider citing the paper as follows:
```
@article{amanda2021gailloss,
  title={Experiments on GAIL loss in Imitation Learning},
  author={Amanda Dsouza},
  journal={arXiv preprint arXiv:TBD},
  year={2022}
}
```
