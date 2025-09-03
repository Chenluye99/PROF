<div align="center">

# Beyond Correctness: Harmonizing Process and Outcome Rewards through RL Training

</div>

## Table of Contents
- [Introduction](#introduction)


## Introduction
### Current Dilemma: Process vs. Outcome

Current Reinforcement learning with verifiable rewards (RLVR) faces a challenge:

- Outcome Reward Models (ORMs): They are "results-oriented," caring only whether the final answer is correct. This is reliable but also "blind"—they cannot distinguish between a logically perfect solution and one with flawed reasoning steps but happens to be correct.

- Process Reward Models (PRMs): They are "process-oriented" and can evaluate each step of the reasoning. This provides finer-grained guidance, but PRMs themselves are often inaccurate and susceptible to "reward hacking" by the model.

Simply blending the two often introduces unstable training signals and can even lead to worse performance.

### Our Solution: PROF - PRocess cOnsistency Filter

We propose **PRocess cOnsistency Filter (PROF)**, an effective data process curation method that the strengths of process rewards (PRMs) and outcome rewards (ORMs). The core idea of PROF is not to use the PRM's score directly for training, but rather to use it as a "consistency filter" to select the highest-quality training data. Specifically, PROF filters the correct and incorrect samples separately and remove the most deceptive "noise":

- Correct: remove the responses with the lowest PRM values averaged over steps
- Incorrect: remove  the responses with the highest PRM values averaged over steps

The number to remove in correct and incorrect group is to balance the correct-incorrect ratio.

<p align="center">
  <img src="fig/PROF_alg.jpeg" width="72%" />
</p>

<p align="center">
  <img src="fig/Qwen2.5-Math-1.5B_data_temp1.0_comparison_main.png" width="45%" />
  <img src="fig/7B_model_comparison_temp1.0_main.png" width="45%" />
</p>

### Main Takeways:
1. 
