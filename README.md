# Conditional Mutual Information for Disentanglement

This is the implementation of Conditional Mutual Information for Disentanglement (CMID) from the paper 
Conditional Mutual Information for Disentangled Representations in Reinforcement Learning.

This code is based on the DrQ PyTorch implementation by [Yarats et al.](https://github.com/denisyarats/drq) 
and the DMControl Generalisation Benchmark by [Hansen et al.](https://github.com/nicklashansen/dmcontrol-generalization-benchmark)
which also contains the official SVEA implementation. As per the original code bases, 
we use [kornia](https://github.com/kornia/kornia) for data augmentation.

The CMID auxiliary task applied to SVEA as the base RL algorithm is largely contained in the `algorithms/svea_cmid.py` 
file. The `dmc2gym` folder contains the [dmc2gym](https://github.com/denisyarats/dmc2gym) code amended slighty to create the colour correlations.

## Requirements
We assume you have access to [MuJoCo](https://github.com/openai/mujoco-py) and a GPU that can run CUDA 11.7. 
Then, the simplest way to install all required dependencies is to create a conda environment by running:
```(python)
conda env create -f conda_env.yml
```
You can activate your environment with:
```(python)
conda activate cmid
```

## Instructions
You can run the code uing the configuration specified in `arguments.py` with:
```(python)
python train.py
```

The `configs` folder contains bash scripts for all the algorithms used in the paper 
on the cartpole task as an example. You can run a specific configuration using the 
bash script, for example:
```(python)
sh configs/cartpole_colour_correlation_svea_cmid.sh
```

This will produce the `runs` folder, where all the outputs are going to be stored including train/eval logs.


The console output is also available in the form:
```
| train | E: 5 | S: 5000 | R: 11.4359 | D: 66.8 s | BR: 0.0581 | ALOSS: -1.0640 | CLOSS: 0.0996 | TLOSS: -23.1683 | TVAL: 0.0945 | AENT: 3.8132 | CMIDD: 0.7837 | CMIDA: 0.6953
```
a training entry decodes as
```
train - training episode
E - total number of episodes
S - total number of environment steps
R - episode return
D - duration in seconds
BR - average reward of a sampled batch
ALOSS - average loss of the actor
CLOSS - average loss of the critic
TLOSS - average loss of the temperature parameter
TVAL - the value of temperature
AENT - the actor's entropy
CMIDD - average of the CMID discriminator loss
CMIDA - average of the CMID adversarial loss
```
while an evaluation entry
```
| eval  | E: 20 | S: 20000 | R: 10.9356
```
contains
```
E - evaluation was performed after E episodes
S - evaluation was performed after S environment steps
R - average episode return computed over `num_eval_episodes` (usually 10)
```
