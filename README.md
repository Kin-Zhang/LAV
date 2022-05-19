# LAV
原repo地址：[https://github.com/dotchen/LAV](https://github.com/dotchen/LAV)

原paper: [Learning from all vehicles](http://arxiv.org/abs/2203.11934)

此处仅作为参考对比代码，更多讨论见discussion部分！Please click discussion section to discussion on this task.

## Getting Started
* To run CARLA and train the models, make sure you are using a machine with **at least** a mid-end GPU.

* Please follow [INSTALL.md](docs/INSTALL.md) to setup the environment.

  1. clone 与 git_lfs下载

      ```bash
      git clone --recurse-submodules git@github.com:Kin-Zhang/LAV.git
      curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
      ```

  2. 环境 与 [CUDA安装 11.3](https://blog.csdn.net/qq_39537898/article/details/120928365#t5)

     ```bash
     conda env create -f environment.yaml
     conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
     conda install pytorch-scatter -c pyg
     ```

* We also release our LAV dataset. Download the dataset [HERE](https://utexas.box.com/s/evo96v5md4r8nooma3z17kcnfjzp2wed). [还没有开数据集是如何收集的，本来以为不太多 打算试试，然后一看 emmm 太多label了 还是等作者开把... ]

## Training
We adopt a LBC-style staged privileged distillation framework.
Please refer to [TRAINING.md](docs/TRAINING.md) for more details.

以下为简单版复制阶段：

1. Privileged Motion Planning

   ```bash
   python -m lav.train_bev
   ```

2. Semantic Segmentation

   ```bash
   python -m lav.train_seg
   ```

3. RGB Braking Prediction

   ```bash
   python -m lav.train_bra
   ```

4. Point Painting, 主要是针对前面训出来的 对数据集进行添加

   ```bash
   python -m lav.data_paint
   ```

5. Perception Pre-training

   ```bash
   python -m lav.train_full --perceive-only
   ```

6. End-to-end Training

   ```bash
   python -m lav.train_full
   ```

   

## Evaluation
We additionally provide examplery trained weights in the `weights` folder if you would like to directly evaluate. They are trained on Town01, 03, 04, 06. Make sure you are launching CARLA with the `-vulkan` flag.


运行 `./leaderboard/scripts/run_evaluation.sh` 其中文件可改为此处
```bash
#!/bin/bash

#!改这两个地址=====
export CARLA_ROOT=/home/kin/CARLA
export LAV=/home/kin/lav

export LEADERBOARD_ROOT=${LAV}/leaderboard
export SCENARIO_RUNNER_ROOT=${LAV}/scenario_runner
export PYTHONPATH=$PYTHONPATH:"${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${CARLA_ROOT}/CARLA/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
export TEAM_AGENT=${LAV}/team_code/lav_agent.py
export TEAM_CONFIG=${LAV}/team_code/config.yaml

export SCENARIOS=${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json
export ROUTES=${LEADERBOARD_ROOT}/data/routes_devtest.xml
export REPETITIONS=1
export CHECKPOINT_ENDPOINT=results.json
export DEBUG_CHALLENGE=0
export CHALLENGE_TRACK_CODENAME=SENSORS

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME}
```
Use `ROUTES=assets/routes_lav_valid.xml` to run our ablation routes, or `ROUTES=leaderboard/data/routes_valid.xml` for the validation routes provided by leaderboard.

## Acknowledgements
We thank Tianwei Yin for the pillar generation code.
The ERFNet codes are taken from the official ERFNet repo.

## License
This repo is released under the Apache 2.0 License (please refer to the LICENSE file for details).
