---
layout: post
title:  "Comparing verbosity bias in Human and LLM based evaluators"
date:   2023-07-26 12:32:00 -0700
categories: Large Language Models, Evaluation, Bias
permalink: verbosity-bias-llms.html
---


Much has been reported recently about biases in LLM based evaluators - for instance, LLM’s have been reported to exhibit position bias favoring options in earlier positions over others [[1]](#references) [[2]](#references) [[3]](#references), verbosity bias favoring more verbose outputs [[4]](#references), diversity bias favoring outputs using higher number of distinct tokens, and bias towards LLM-based outputs [[5]](#references). These have often been used as a critique of such approaches. In addition, the most common method of validating automated metrics is their correlation with human judgement. It’s worthwhile, then, to ask the question: Do humans suffer from the same biases?

In this micro-post, I dig into the verbosity bias on two benchmarks: The first, a task of miscellaneous prompts from the MT-bench [[4]](#references) dataset. MT-bench contains 3.3K pairwise human preferences on model outputs from 6 strongly performing large language models - GPT-4, GPT-3.5, Claud-v1, Vicuna-13B, Alpaca-13B, and LLaMA-13B, which have been annotated for preference by graduate students with expertise on 8 topics (writing, roleplay, extraction, reasoning, math, coding, STEM, humanities/social science). A second, a news summarization benchmark, using a dataset of preferences over LLM generated summaries and high-quality summaries for news articles written by freelance writers [[6]](#references). A reasonable hypothesis is that while humans (and LLMs) might lean towards verbosity in descriptive tasks, brevity is a reasonable preference on a summarization task.

### Experiments exploring verbosity bias in human and LLM evals

LLM’s have been found to favor longer, more verbose responses, compared to shorter alternatives.

In order to examine verbosity bias, we first tokenize (by simply splitting-on-space) the text, and compute the length of tokens in the winning generated outputs. In cases of a tie in the evaluation, we randomly select one of the model outputs as winner for computing its token length. A binary flag is set to record when a longer winning token length is preferred over a losing token length.

#### On MT-Bench prompts

We compare the model generated outputs in MT-bench evaluated by human experts and GPT-4, and find that the rate of preferring longer generated text outputs is nearly the same by both, while being slightly higher for GPT-4 *(Human eval - 61.22% versus GPT-4 eval - 65.95%)*. In most cases, it is natural to assume that longer generations convey more information or are perceived as better, but this may not necessarily be accurate.

<p align="center" style="font-size:8px;">
<img src="https://amy12xx.github.io/img/verbosity_bias/human-gpt4-eval.png" width=360>
</p>

Between outputs that were marked as ties (between the two competing models), and those that were not ties (strict preference of one or the other model), I found that ties had a lower average token length difference between the model outputs, but that this behavior was similar to both human and LLM evaluations. 

<p align="center" style="font-size:8px;">
<img src="https://amy12xx.github.io/img/verbosity_bias/mean token len.png" width=360>
</p>

When I described verbosity bias above, I left out some parts that are used in its definition in the MT-bench paper - *“even if they are not as clear, high-quality, or accurate”,* mainly because I think the inductive bias towards longer outputs itself may exist without this distinction, and more importantly, because I do not think there have been sufficient studies to show that this is the case.

For example, MT-Bench creates an artificial “repetitive attack” task where a list of *n* items containing the answers are extended to *2n* by paraphrasing the existing *n* items, artificially inflating it. Only GPT-4 is capable of sustaining this attack, while Claude-v1 and GPT-3.5 favor the longer list outputs. While this task is an interesting one, it is not reflective of a typical output of current strongly performing models, and is perhaps less useful in a real-world benchmark.

Intuitively, it’s easy to see how descriptive and creative tasks such as *“Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions”* or *“Describe a vivid and unique character, using strong imagery and creative language. Please answer in fewer than two paragraphs”* would prefer longer outputs over more concise ones.

### On a News Summarization task

We look at the task of news summarization, where, one would expect a preference of concise outputs over more verbose ones. However, on the news summarization dataset, we find that the same pattern as above holds, with longer “overall” summaries being preferred over shorter ones (Human eval - 60.76%), even when the task is to prefer an “overall better summary” (i.e. when not judging for “a more informative summary”).

<p align="center" style="font-size:8px;">
<img src="https://amy12xx.github.io/img/verbosity_bias/summ_percent.png" width=340>
</p>

<p align="center" style="font-size:8px;">
<img src="https://amy12xx.github.io/img/verbosity_bias/summ_mean_token_len.png" width=340>
</p>

### Discussion and Conclusion

If such known biases exist in human evaluations, it begs the question if the underlying task creation needs to be adjusted to first debias it. For instance, on both the general and summarization tasks, could they have included specific instructions that guide the annotators against these known biases, or if the outputs were first normalized or bounded for length. 

Human evaluations have been used as the gold standard on most NLP tasks, including recent benchmarks on LLMs. They have additionally been used in the creation of new metrics (including metrics using LLMs (a.k.a GPT-scorers), where a high correlation of a metric with human judgement is used to validate its reliability as a metric. 

Even in the most basic case, as more and more studies are being published on the effectiveness (or non-effectiveness) of LLM-evaluators, analyzing their biases against human evaluators may help us understand where they come from.


If you use this post in your research, please consider citing as follows:

```
@article{amanda2023verbositybiasllms,
  title={Comparing verbosity bias in Human and LLM based evaluators},
  author={Amanda Dsouza},
  year={2023}
}
```
## References
[1] Miyoung Ko, Jinhyuk Lee, Hyunjae Kim, Gangwoo Kim, and Jaewoo Kang. Look at the first sentence: Position bias in question answering. arXiv preprint arXiv:2004.14602, 2020.

[2] Xuanhui Wang, Nadav Golbandi, Michael Bendersky, Donald Metzler, and Marc Najork. Position bias estimation for unbiased learning to rank in personal search. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pages 610–618, 2018.

[3] Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. Large Language Models are not Fair Evaluators. arXiv preprint arXiv:2305.17926, 2023.

[4] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. arXiv preprint arXiv:2306.05685, 2023.

[5] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu and Chenguang Zhu. G-EVAL: NLG Evaluation using GPT-4 with Better Human Alignment. arXiv preprint arXiv:2303.16634, 2023.

[6] Tianyi Zhang, Faisal Ladhak, Esin Durmus, Percy Liang, Kathleen McKeown and Tatsunori B. Hashimoto. Benchmarking Large Language Models for News Summarization. arXiv preprint arXiv:2301.13848, 2023.