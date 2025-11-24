# TAPPS: Learning Diverse Policies via Task-Adaptive Partial Parameter Sharing in Heterogeneous Multi-Agent Reinforcement Learning

This repository contains the implementation of **TAPPS**, a novel multi-agent reinforcement learning algorithm that enables task-adaptive partial parameter sharing for learning diverse policies in heterogeneous multi-agent systems.
TAPPS is built as an extension of [EPyMARL](https://github.com/uoe-agents/epymarl), which itself extends [PyMARL](https://github.com/oxwhirl/pymarl).

## Key Features

- **Task-Adaptive Partial Parameter Sharing**: Dynamically adjusts parameter sharing patterns based on task requirements and agent heterogeneity
- **Heterogeneous Agent Support**: Effectively handles agents with different capabilities and objectives
- **Diverse Policy Learning**: Enables learning of specialized policies while maintaining beneficial parameter sharing
- **Compatibility**: Fully compatible with EPyMARL's environment support and algorithm implementations

## TAPPS Framework Features

TAPPS inherits all EPyMARL features:
- Additional algorithms (IA2C, IPPO, MADDPG, MAA2C and MAPPO)
- Support for [Gymnasium](https://gymnasium.farama.org/index.html) environments (on top of the existing SMAC support)
- Option for no-parameter sharing between agents
- Flexibility with extra implementation details
- Consistency of implementations between different algorithms

# Installation & Run instructions

For information on installing and using this codebase with SMAC, we suggest visiting and reading the original [PyMARL](https://github.com/oxwhirl/pymarl) README. Here, we maintain information on using the extra features EPyMARL offers.
To install the codebase, clone this repo and install the `requirements.txt`.  

## Installing LBF, MPE, and SMAC

In [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/abs/2006.07869) we introduce and benchmark algorithms in Level-Based Foraging, Multi-Robot Warehouse and Multi-agent Particle environments.
To install these please visit:
- [Level Based Foraging](https://github.com/uoe-agents/lb-foraging) or install with `pip install lbforaging`
- [MPE](https://github.com/semitable/multiagent-particle-envs), clone it and install it with `pip install -e .`
- [SMAC](https://github.com/oxwhirl/smacv2).

Example of using LBF:
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="lbforaging:Foraging-8x8-2p-3f-v1"
```
Example of using SMAC:
```sh
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m
```

For MPE, our fork is needed. Essentially all it does (other than fixing some gym compatibility issues) is i) registering the environments with the gym interface when imported as a package and ii) correctly seeding the environments iii) makes the action space compatible with Gym (I think MPE originally does a weird one-hot encoding of the actions).

The environments names in MPE are:
```
...
    "multi_speaker_listener": "MultiSpeakerListener-v0",
    "simple_adversary": "SimpleAdversary-v0",
    "simple_crypto": "SimpleCrypto-v0",
    "simple_push": "SimplePush-v0",
    "simple_reference": "SimpleReference-v0",
    "simple_speaker_listener": "SimpleSpeakerListener-v0",
    "simple_spread": "SimpleSpread-v0",
    "simple_tag": "SimpleTag-v0",
    "simple_world_comm": "SimpleWorldComm-v0",
...
```
Example of using MPE:
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="simple_adversary": "SimpleAdversary-v0"
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

# Run a hyperparameter search

We include a script named `search.py` which reads a search configuration file (e.g. the included `search.config.example.yaml`) and runs a hyperparameter search in one or more tasks. The script can be run using
```shell
python search.py run --config=search.config.example.yaml --seeds 5 locally
```
In a cluster environment where one run should go to a single process, it can also be called in a batch script like:
```shell
python search.py run --config=search.config.example.yaml --seeds 5 single 1
```
where the 1 is an index to the particular hyperparameter configuration and can take values from 1 to the number of different combinations.

# Saving and loading learnt models

## Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

## Loading models

Learnt models can be loaded using the `checkpoint_path` and `load_step` parameters. `checkpoint_path` should point to a directory stored for a run by epymarl as stated above. The pointed-to directory should contain sub-directories for various timesteps at which checkpoints were stored. If `load_step` is not provided (by default `load_step=0`) then the last checkpoint of the pointed-to run is loaded. Otherwise the checkpoint of the closest timestep to `load_step` will be loaded. After loading, the learning will proceed from the corresponding timestep.

To only evaluate loaded models without any training, set the `checkpoint_path` and `load_step` parameters accordingly for the loading, and additionally set `evaluate=True`. Then, the loaded checkpoint will be evaluated for `test_nepisode` episodes before terminating the run.
