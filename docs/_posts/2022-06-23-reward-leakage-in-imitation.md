---
layout: post
title:  "Reward Leakage in Imitation Learning"
date:   2022-06-24 15:32:00 -0700
categories: RL
permalink: reward-leakage.html
---


# Reward Leakage in Imitation Learning


At the heart of Reinforcement Learning, is learning from evaluative feedback. This is typically specified in terms of a reward or cost function. In the absence of a reward function, an agent can learn from an expert (policy or demonstrations), either in a supervised manner (Behavioral Cloning), by first recovering a reward function that is optimal for the demonstrations (Inverse RL), or by learning to generate expert-like trajectories (Adversarial Imitation Learning). Collectively, these methods are considered Imitation Learning (IL). A natural assumption in IL is that the environment provides no feedback to the agent. However, the manner in which most RL environments and sometimes RL libraries are implemented currently, leaves room for leakage of reward information. This problem of reward leakage can be generally observed in the sparse reward setting, and is described below.

Consider a gridworld, where the agent, starting at some position, must navigate to a goal state. An episode terminates when the goal is reached, or some maximum number of timesteps are taken. For instance, in OpenAI Gym, this is usually represented by the "done" flag, and additional "info" buffer. Whatever the reason for termination, when an episode terminates, the "done" flag is set to true. Even without the knowledge of rewards, we can infer from this information. Shorter episodes in this environment are optimal, while longer ones are not. 

The "info" buffer can also leak reward information. Gym, for instance, passes "TimeLimit.truncated" flag, that specifies if the episode terminated due to a terminating state, or a timeout. An example of the popular Stable Baselines 3 shows how the rollout buffer of an on-policy algorithm uses this information to process rewards of terminal states. Gym-robotics environments record a "is_success" flag in the info buffer, when the desired goal is achieved. In general, this information can be exploited by a biased reward function.
```
            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
```

In Adversarial Imitiation Learning algorithms, such as [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476) (GAIL) and its variants, it is not uncommon to use different forms of the reward/cost functions, based on the environment. A reward function ``$\log D(s,a)$`` encourages the agent to terminate quickly by providing negative rewards; good for maze/grid environments where the goal is to complete the task in the fewest number of steps. On the other hand, a reward function ``$-\log (1-D(s,a))$`` acts as a survival bonus, encouraging the agent to stay in the environment to maximize its return; good for environments such as Pong or Breakout. 

These aren't the only settings. Consider a gridworld environment with additonal negative rewards or penalty, for instance, the toy frozen lake environment. In this case, either reward function does not bias towards the desired behavior. Consequently, such environments may be less influenced by the choice of reward sign. Still other environments may be more complex, for which this reward bias is not sufficient for learning. Nevertheless, it is common practice to use one of these rewards forms as priors in the imitation learning paradigm.

The problem of variable length episodes has been well described [here](https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html). And the effects of using a biased reward function have been explained by [Kostrikov et al](https://arxiv.org/abs/1809.02925).

In this article, we demonstrate how a pathological reward function can learn even in more complex settings of learning from visual observations on several gridworld and 3D tasks, and show much better performance than the baseline [GAILfO](https://arxiv.org/abs/1807.06158) (GAIL from observations). 


## Experiment Details

<p align="center" style="font-size:8px;">
<img src="https://amy12xx.github.io/img/reward_leakage/gail.png" width=360></img>
</br>
</p>

GAIL uses adversarial learning to jointly train a discriminator and generator on a state-action occupancy matching objective. The discriminator loss, is similar to the GAN discriminator loss, optimizing a dual objective; i.e. increasing the cost of non-expert demonstrations, while decreasing the cost of expert demonstrations.

<p align="center" style="font-size:8px;">
<img src="https://amy12xx.github.io/img/reward_leakage/gailfo_loss.png" width=360>
<img src="https://amy12xx.github.io/img/reward_leakage/gailfo_reward.png" width=160>
</p>

We create a pathological loss function, by removing the loss component that optimizes the cost of expert demonstrations. In this case, the expert demonstrations are no longer used for training, and the result is a network that predicts a constant negative reward for all input observations. Minimizing the loss during training results in reducing the penalty given to the agent. Theoretically, any constant (unchanged during training) reward function, should still let the agent learn the optimal behaviour in sparse reward environments with reward leakage (since this is equivalent to scaling of the reward). However, in our experiments with several choices of constant rewards, we do not find this to be true. It could be a combination of different factors, such as complexity of learning from visual observations, insufficient tuning, choice of discount factor etc. Nevertheless, the "reward function" we use provides no distinguishing signal to the agent. Learning then occurs, due to leakage from the terminal/non-terminal information and optimizing shorter trajectories due to the negative rewards.

<p align="center" style="font-size:8px;">
<img src="https://amy12xx.github.io/img/reward_leakage/gailfo_reward2.png" width=160>
</p>

## Experiment Results

### On sparse reward tasks

We conduct experiments using GAILfO as the baseline, and our pathological loss, which we refer to as "agent", since it only uses agent data in the loss computation. To test the sparse reward setting, we use the [Minigrid](https://github.com/Farama-Foundation/gym-minigrid) (minimalistic gridworlds), and [Miniworld](https://github.com/Farama-Foundation/gym-miniworld) (minimalistic 3D environment, similar to DM Lab) tasks. We use pixel observations, with no action information. Some of these environment require the agent to simply reach the end state, while others require it to take some action, such as picking up a key, in order to achieve the goal of opening a door. We use stacked observations for Miniworld, since it is a more difficult environment than gridworld. As an example, an expert policy for the TMazeRight or TMazeLeft environments, on average, takes 60 timesteps to reach the goal at the end of a T shaped maze. In all these tasks, there are no negative absorbing states. 


<p align="center" style="font-size:8px;">
<img src="https://amy12xx.github.io/img/reward_leakage/minigrid and miniworld gail loss.jpg">
Mean Reward + 95% confidence levels on Minigrid, and Miniworld environments, using pixel observations (egocentric observations in case of Miniworld), over 3 seeds.
</p>

In contrast, the Lava gap task has both positive and negative absorbing states. An agent must reach the goal state, while avoiding the lava. In this case, the agent loss, does not learn, since no single biased reward function allows it to terminate quickly while avoiding the lava states.

<p align="center" style="font-size:8px;">
<img src="https://amy12xx.github.io/img/reward_leakage/lava.jpg">
W&B sweeps on the Lava environment.
</p>


### On dense reward tasks

Similar to the Lava case, we find that dense reward tasks are also less sensitive to the reward leakage. Since the two cases of terminating an episode - either by executing the maximum number of steps, or reaching a goal - require the agent to stay in the environment for the full length of the episode, these two cases are indistinguishable, and therefore, difficult to exploit. For the dense reward settings, we test using stacked pixel observations, on Mujoco tasks (Walker and Hopper) and using (state-action) demonstrations, on classic control tasks (CartPole, Acrobot and Pendulum).

<p align="center" style="font-size:8px;">
<img src="https://amy12xx.github.io/img/reward_leakage/classic.jpg">
Results on dense reward tasks: Classic Control tasks (CartPole, Acrobot and Pendulum) using demonstrations (LfD) and Mujoco using pixel observations, over 3 seeds.
</p>



The problem of reward leakage can inadvertently occur even in more complex environments. However, it can be easily alleviated by modifying the environments to stop the reward leakage. For instance, the seals project provides fixed horizon versions of some Gym environments, by always returning ``done=False`` and ``info={}``. Model diagnoses (for instance, through tensorboard/W&B logs) and test cases are good places to catch anomalies, in general.



If you use this post in your research, please consider citing as follows:

```
@article{amanda2021rewardleakage,
  title={Reward Leakage in Imitation Learning},
  author={Amanda Dsouza},
  year={2022}
}
```
