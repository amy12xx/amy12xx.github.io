---
layout: post
title:  "Examining Gradients of Generalization in RL"
date:   2022-08-10 17:28:00 -0700
categories: Reinforcement Learning, Psychology, Cognitive Science
permalink: generalization-gradients.html
---


# Generalization in scientific fields
 One of the hallmarks of any learning system is Generalization. Loosely defined as the capability to learn general concepts from specific training examples, it is something that comes very naturally to human beings. From only a few examples, human beings are capable of applying the learning to a wide spectrum of similar and sometimes unrelated tasks. How (and how much) we generalize can have significant effects on well being. It is no wonder then, that much effort has been spent in understanding it, in various fields of science. In Neuroscience, studies of memory representations [[1]](#references) and instabilities [[2]](#references) that support generalization, environmental complexity that governs the degree of generalization in spatial movement [[3]](#references) have tried to advance our understanding of *why* generalization occurs. In Psychology, generalization to different stimuli (for example, fear) have been studied extensively, and have tried to advance our understanding of *how* generalization occurs.


In our quest to create intelligent agents, where intelligence has been, so far, benchmarked by human cognitive abilities, one of the fundamental goals of Machine Learning is Generalization. To that end, extensive research has been carried out in defining Generalization [[4]](#references) [[5]](#references) [[6]](#references), and towards building generalizable agents and models [[7]](#references), [[8]](#references), [[9]](#references). In the large deep learning models era, studying Generalization has taken on a new focus. This has also extended to Reinforcement Learning, with several studies of the problem [[10]](#references) [[11]](#references) [[12]](#references), new benchmarks proposed [[13]](#references) [[14]](#references) [[15]](#references), and algorithms developed.

However, (from a limited search) there is little literature of Generalization in both Machine Learning and Reinforcement Learning, that strongly draws from the aforementioned fields of science. In an attempt to bridge that gap, I take a look at one important area related to Generalization in Psychology, namely "Gradients of Generalization", validating it on (a small subset of) RL agents. In a small sample study, I find evidence that RL agents may follow the universal law of Generalization, under specific learning conditions. If this is found to be true, we could perhaps transfer some of the well studied findings in animal pyschology (such as peak shifts [[16]](#references), avoidance learning [[17]](#references)) to artifical agents, for instance to construct intentional generalization behavior. This could potentially have implications in fields of AI Safety and Robustness. Further, perhaps gradients of generalization could be used to determine generalization guarantees, similar to scaling laws.

### Conditioned reflexes

We start by briefly reviewing some of the early work in classic and operant conditioning in Psychology. In 1927, Pavlov, in his seminal experiments on classic conditioning [[18]](#references), showed that a neutral stimulus (bell sound) associated with an unconditioned stimulus (food) was used to generate a conditioned response (salivating at the bell sound) in dogs. This experiment is well known today. In what is perhaps less well known, subsequent experiments by Pavlov showed that conditioned responses were found to occur on test stimuli different, but similar (in pitch, for instance) to the original conditioned (training) stimulus. This led to numerous experiments [[19]](#references), analyzing "gradients of stimulus generalization", measuring the degree of learnt responses to distances between the test and original training stimulus.

### Univeral law of Generalization

<p align="center" style="font-size:8px;">
<img src="https://amy12xx.github.io/img/theory_of_gen/universal_law.png" width=360>
</p>
<p align="center" style="font-size:8px;">
12 Gradients of Generalization following exponential decay curves, Roger Shepard, 1987 (Source: [20])
</p>

In 1987, following this work, Roger Shepard showed that there exists a universal law of generalization [[20]](#references), which is the focus of these experiments. According to the law, the probability to which a learnt response to a specific stimulus generalizes to another different stimulus depends on the "distance" between the stimuli and follows an exponential decay with this distance. Importantly, this distance measure is not in physical space, but one in psychological space.

Previous attempts at measuring "gradients of generalization" used physical measures of differences between stimuli, (for example, frequency / size / wavelengths). Even though physical differences lead to an overall decrease of generalization with increasing distance, the decrease is not necessarily monotonic nor invariant. Shepard, therefore, sought to find a monotonic and invariant function, whose inverse transformed the observed generalization data into distances in some space (he termed as psychological space). Psychological space can be understood as equivalent to latent space in Machine Learning. Shepard found that the exponential decrease in this "psychological space" follows universally among different stimuli, sensory modalities, and across multiple species.

### Mathematical formulation

Shepard proposes the following method to uncover the universal law: For generalization data G, on a set of stimuli *S* (s+ as the original trained stimuli), and a distance metric *d*, we want to recover a function *f* as,

<p align="center" style="font-size:8px;">
<img src="https://amy12xx.github.io/img/theory_of_gen/formula.png" width=360>
</p>

This can be done by using non-metric multidimensional scaling (NMDS) [[21]](#references). NMDS finds a lower dimensional space which preserves the ordering of the similarity in data. It models the similarity in the data as distances in metric space. On an *n X n* symmetric matrix of (normalized) generalization measures *g_ij*, NMDS finds a lower dimensional space in some *k (k << n)* dimensions. Points in this space can now represent distances between the original points and are invariant to the underlying experiment data. Then, using some metric distance (Shepard shows that Euclidean and Manhattan distances work well for stimuli data), one can uncover the universal law.


## Experiment details

Shepard's experiments are conducted on humans and animals on wide range of stimuli such as colors - lightness, saturation, sizes, spectral hues, phonemes, shapes, and morse code signals. Subsequent studies [[22]](#references) conducted experiments on distance and direction stimuli. In order to replicate the work, I conduct experiments on three simple environments, OpenAI Gym [[23](#references) Classic Control, MiniGrid [[24]](#references) and MiniWorld [[25]](#references). The aim is to examine what the gradients of generalization of RL agents look like for different "stimuli" (interpreted as factors or parameters of the agent or environment), and to test the hypothesis if RL agents follow a universal law of generalization. In order to do this, the following stimuli are varied in separate experiments: hue, saturation, orientation, distance of the goal object and length of agent.

MiniGrid random grids are simple grid layouts with the agent initialized to a random position. On the 6x6 and 8x8 random grids, we vary the hue and saturation of the goal tile. In order to avoid overfitting of the agent to the goal position, we randomize the goal position. MiniWorld is a Minimalistic 3D environment with egocentric view observations and discrete actions. In the Hallway environment, a box (or object) is placed at the far end of the hallway, with the agent at the other end, with the goal of the agent to pick up the box. We conduct experiments similar to MiniGrid, varying the hue (9 discrete hues from red to green) and saturation (from 0% to 100%). Pixel observations are used on these experiments. An additional experiment varies the orientation of a large key as the goal (from 0 to 180 degrees), with grayscale image observations to avoid the agent overfitting to the color attributes of the goal. Lastly on Classic Control, I use Cartpole with varying lengths of poles, and Lunar Lander, with varying positions of the lander to the landing pad, in the range [-5,5]. In order to control for difficulty of the task at different landing pad positions, the terrain is made smooth everywhere, and the lander starts exactly above the landing pad, in all cases, similar to the default setting.

PPO is used as the learning algorithm, over 5 random seeds, and each policy is evaluated over 100 episodes. A CNN network is used to learn from visual observation data. To learn a latent space based on the original distances, I use NMDS as proposed in the original paper, on the generalization data, with Euclidean distance in the resultant latent space to compute the generalization curves.

## Experiment results

<p align="center" style="font-size:8px;">
<img src="https://amy12xx.github.io/img/theory_of_gen/combined-neutral-se.png">
Degree of Generalization with latent distance on Hue and Saturation in MiniGrid 6x6, 8x8 and MiniWorld-Hallway environments, averaged over 5 random seeds, and evaluated over 100 episodes. Plots exhibit stretched exponential decays with increasing distance in latent space. 
</p>

In experiments involving hue and saturation, the degree of generalization to latent distance exhibits a stretched exponential curve. (A stretched exponential uses a fractional power law in the exponential function). This implies that to some extent an agent will generalize to varying stimuli, before sharply dropping off. Both hue and saturation can be considered to be spurious correlation features, that are irrelevant to the task. They are also independent along the spectrum they are varied over. For instance, having a red or green box as the object to pick up should not render the problem more or less difficult. As Shepard points out, for stimuli such as color, the generalization is symmetric, since they do not possess any preferred axes in their psychological space (since the dimenions do not correspond to real world independent attributes). However, I am not sure if this claim holds for artificial agents. Regardless, in the language of Psychology, these stimuli can be considered as "neutral" stimuli, which do not affect the behavior of the conditioned stimulus. It appears that this is an important distinction to recover the exponential decay in RL agents. 

<p align="center" style="font-size:8px;">
<img src="https://amy12xx.github.io/img/theory_of_gen/combined-nonneutral-se.png">
Degree of Generalization with latent distance on Distance of lander to landing pad (Left) and Orientation of key (Right), averaged over 5 random seeds, and evaluated over 100 episodes.
</p>

In the case of lunar lander experiments using the distance (between the agent and landing pad) stimulus, I expected to observe a similar generalization curve. All things being equal, one would expect to see generalization sharply drop off the further the landing pad from the agent's starting position, and expectation of goal position. And further, that this would be invariant to the direction (left or right) of generalization. However, the results are somewhat surprising. It appears that the agent is able to generalize more favorably in one direction, than the other. Recall that the terrain has been made smooth throughout. The only other explanation could be the dynamics that cause it to behave this way. Specificically the turbulence power applied to the lander may be biased towards one direction (since wind effect is set to false in the default lander settings).

In the case of orientation experiments, we find that the graph decays more gradually. This implies, that the agent is able to generalize better over the different orientations. This may be due to the limited variation in the orientation stimulus0. Using other distance metrics such as absolute cosine distances does not recover an exponential curve. Given that some orientations (less occluded) ought to be more easily learnable than others, I am uncertain if a stretched exponential can be recovered.

In a similar manner, it is not possible to recover an exponential generalization curve on cartpole lengths. Note that in the case of both orientation and cartpole lengths, the stimuli are not neutral, but instead, change the task (in complexity or otherwise). In which case, it may not be possible to recover the exponential curves using the Euclidean or Manhattan distance, as seen in the above experiments. This has been also elucidated in [[26]](#references), as challenges to the universal law, concerning asymmetry of the (properties of) psychological stimuli.

<p align="center" style="font-size:8px;">
<img src="https://amy12xx.github.io/img/theory_of_gen/raw data plots.png">
Degree of generalization with true distance. (a) Hue on MiniGrid-6x6 grid as a reference of exponentially decaying generalization (b) Distance of lunar lander to landing pad appears to generalize better left-to-right, than right-to-left. (c) Cartpole lengths unsurprisingly exhibit better generalization from long to short pole lengths, than vice-versa (d) Difference of acute angles of key (goal) in Hallway environment exhibit a linear decay with a larger variance. 
</p>


## Discussion and critiques of universal law

While these experiments replicating Shepard's work on RL agents are more of a proof of concept (or curiousity), on some neutral stimuli, we found that evidence that generalization may decay, in a universal manner, following a (stretched) exponential curve. When stimuli are not neutral but affect the learning capacity, we can no longer expect the same decay patterns. Of course, more experiments would be necessary to substantiate any such claim.

Finally, there have been studies challenging aspects of the universal law. Shepard showed that the metric used in uncovering the universal law differed based on the nature of stimuli. In order to alleviate this dependence, [[26]](#references) proposed a more abstract information metric (expressed in terms of Koglomorov complexity) that can work universally on arbitrary complex stimuli. (Several other critiques of the law are expanded on in the paper which we leave to the interested reader). [[27]](#references) suggests that rate distortion theory provides an explanation for the universal law, and that any efficient system operating under constraints of limited capacity (biological or artificial) naturally recovers the exponential decay. Further, that Shepard's law is a specific case of the law dictated by rate distortion. [[28]](#references) shows that the exponential decay is due to the properties (shift and stretch invariance) of the perceptual scale of stimuli, and anything beyond that, such the type of transformation (NMDS) or distance metric (Koglomorov complexity) is merely superfluous.

Inspite of these critiques, studying gradients of generalization in artificial agents could be useful, to understand or apply some of the broad research that has been conducted on psychological studies of this phenomenon.

If you use this post in your research, please consider citing as follows:

```
@article{amanda2022gradientsofgen,
  title={Examining Gradients of Generalization in RL},
  author={Amanda Dsouza},
  year={2022}
}
```
## References
[[1] Caitlin R. Bowman and Dagmar Zeithamova. Abstract Memory Representations in the Ventromedial Prefrontal Cortex and Hippocampus Support Concept Generalization](https://www.jneurosci.org/content/38/10/2605.abstract)

[[2] Edwin M. Robertson. Memory instability as a gateway to generalization](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2004633)

[[3] Kurt A. Thoroughman and Jordan A. Taylor. Rapid Reshaping of Human Motor Generalization](https://www.jneurosci.org/content/25/39/8948.short)

[[4] L G. Valiant. A Theory of the Learnable](http://web.mit.edu/6.435/www/Valiant84.pdf)

[[5] Kenji Kawaguchi, Leslie Pack Kaelbling, Yoshua Bengio. Generalization in Deep Learning](https://arxiv.org/abs/1710.05468)

[[6] David Haussler. Probably Approximately Correct Learning](https://www.aaai.org/Papers/AAAI/1990/AAAI90-163.pdf) 

[[7] Robert E. Schapire. The Boosting Approach to Machine Learning: An Overview](https://link.springer.com/chapter/10.1007/978-0-387-21579-2_9)

[[8] Scott Reed et al. A Generalist Agent](https://arxiv.org/abs/2205.06175)

[[9] Rich Caruana. Multitask Learning](https://link.springer.com/article/10.1023/A:1007379606734)

[[10] Robert Kirk et al. A Survey of Generalisation in Deep Reinforcement Learning](https://arxiv.org/abs/2111.09794)

[[11] Chiyuan Zhang et al. A Study on Overfitting in Deep Reinforcement Learning](https://arxiv.org/abs/1804.06893)

[[12] Charles Packer et al. Assessing Generalization in Deep Reinforcement Learning](https://arxiv.org/abs/1810.12282)

[[13] Alex Nichol et al. Gotta Learn Fast: A New Benchmark for Generalization in RL](https://arxiv.org/abs/1804.03720).

[[14] Karl Cobbe et al. Leveraging Procedural Generation to Benchmark Reinforcement Learning](https://arxiv.org/abs/1912.01588)

[[15] Austin Stone et al. The Distracting Control Suite -- A Challenging Benchmark for Reinforcement Learning from Pixels](https://arxiv.org/abs/2101.02722)

[[16] Edward P. Kardas. Generalization and Peak Shift](http://peace.saumag.edu/faculty/kardas/Images/Web%20Images/Lecture/DiscPeak.html)

[[17] Agnes Norbury, Trevor W Robbins, Ben Seymour. Value generalization in human avoidance learning](https://elifesciences.org/articles/34779)

[[18] P Ivan Pavlov. Conditioned reflexes: An investigation of the physiological activity of the cerebral cortex](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4116985/)

[[19] Norman Guttman and Harry I. Kalish. Discriminability and stimulus generalization](https://psycnet.apa.org/record/1956-06725-001)

[[20] Roger N. Shepard. Toward a Universal Law of Generalization for Psychological Science](https://www.science.org/doi/10.1126/science.3629243)

[[21] J. B. Kruskal. Nonmetric multidimensional scaling: A numerical method](https://link.springer.com/article/10.1007/BF02289694)

[[22] Ken Cheng. Shepard's Universal Law Supported by Honeybees in Spatial Generalization](https://journals.sagepub.com/doi/10.1111/1467-9280.00278)

[[23] Greg Brockman et al. OpenAI Gym](https://arxiv.org/abs/1606.01540)

[[24] Chevalier-Boisvert et al. Minimalistic Gridworld Environment for OpenAI Gym](https://github.com/Farama-Foundation/gym-minigrid)

[[25] Maxime Chevalier-Boisvert. MiniWorld: Minimalistic 3D Environment for RL & Robotics Research](https://github.com/Farama-Foundation/gym-miniworld)

[[26] Nick Chater and Paul M. B. Vitanyi. The generalized universal law of generalization](https://www.sciencedirect.com/science/article/abs/pii/S0022249603000130)

[[27] Chris R. Sims. Efficient coding explains the universal law of generalization in human perception](https://www.science.org/doi/10.1126/science.aaq1118)

[[28] Steven A. Frank. Measurement invariance explains the universal law of generalization for psychological perception](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6166795/)