# DI-bioseq

<img src="./docs/figs/di-bioseq_banner.png" alt="icon"/>

Updated on 2022.4.24

DI-bioseq - Decision Intelligence platform for biological sequence prediction and searching.

## Introduction

**DI-bioseq** is an open-source application platform under [**OpenDILab**](http://opendilab.org/). DI-bioseq provides Reinforcement Learning pipeline for **biological sequence searching**, including DNA, RNA, and amino acid sequences. Biological sequence searching problems aims to find a sequence with a fixed length of these elements that has the best characteristics that humans desired, represented by a score. The search starts from a series of known sequences whose scores are measured via biological experiments. The searching procedure is expected to be high-efficient, with less scores referred to from biological experiments, rather than trying a lot to find the best one.

Biological sequence searching pipelines often consists of the following modules:

- A biological ground-truth data/simulator that can get the fitness score of a proposed sequence, always called **Landscape**
- A prediction **Model** that attempts to fit the ground-truth landscape from known score
- An **Encoder** to convert a sequence into numbers
- An **Engine** which is the core method to find new sequence proposals from known ones and scores

**DI-bioseq** uses [**DI-engine**](https://github.com/opendilab/DI-engine), an RL platform to build searching methods. The environment is set to generate new sequences from existing ones, and the predicted score from model together with some other statistics is used as reward.

**DI-bioseq** was developed referencing [FLEXS](https://github.com/samsinai/FLEXS) on biological environments and modules.

## Guidence

### Installation

**DI-bioseq** together with **DI-engine** can be easily installed from source via `pip`.

```bash
git clone https://github.com/opendilab/DI-bioseq.git
cd di-bioseq
pip install -e . --user
```

### Features

**DI-bioseq** currently supports the following searching modules.

|  Landscape   | Engine  |  Model  |  Encoder  |  
|  ----  | ----  |  ----  |  ----  |
| GB1  |  Random |  Linear Regression  |   Onehot  |
| TF-Binding  |  Off-policy PPO(onehot only)  |    Random Forest Regression   |   Georgiev(protein only)   |

### Quick start

**DI-bioseq** provides a standard searching pipeline and entry that can be configured 

```bash
main.py [-h] [--landscape LANDSCAPE] [--engine ENGINE] [--model MODEL]
               [--encoder ENCODER] [--score SCORE] [--predict PREDICT]
               [--round ROUND]

optional arguments:
  -h, --help            show this help message and exit
  --landscape LANDSCAPE
                        landscape type
  --engine ENGINE       searching engine type
  --model MODEL         prediction model type
  --encoder ENCODER     encoder type
  --score SCORE         max score in landscape each round
  --predict PREDICT     max prediction in model each round
  --round ROUND         searching rounds
```

## Contributing

We appreciate all contributions to improve DI-bioseq, both algorithms and system designs.

## License

DI-bioseq released under the Apache 2.0 license.

## Citation

@misc{bioseq,
    title={{DI-bioseq: OpenDILab} Decision Intelligence platform for biological sequence prediction and searching},
    author={DI-bioseq Contributors},
    publisher = {GitHub},
    howpublished = {\url{`https://github.com/opendilab/DI-bioseq`}},
    year={2021},
}
