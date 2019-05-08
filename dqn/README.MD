# Reinforcement Learning Project: Super Mario Brothers 3
Uppsala University: Machine Learning Course Project

## Notes
Please be aware that this implementation currently still uses `Tensorflow 1.3` and not `2.0`. The goal is to convert this in a future version!

## Install
- `python ./install/setup.py` (installation including conda environment)
  - If it doesn't work, `python -m pip install -r ./install/requirements.txt` should work. 
- Try if it worked by running:
    - `python -m retro.examples.interactive --game SuperMarioBros3-Nes`

## Rewards
Have a look at the files: `data.json` and `scenario.json` to see how the rewards are set. 
- `scenario.json` is responsible for the rewards returned in the object `reward` returned by `env.step()`. 
- A custom reward function can be found in `agent.py`

## Hyperparameters
The different hyperparameters and configurations can be found in: `config.py`

## Replay
To replay a training result. Run `python replay.py`, beforehand adjust the path to the recording location in `config.py`.

### Converting the best result to MP4
- Requires [ffmpeg](https://ffmpeg.org/download.html) to be installed. 
  - Download and install it beforehand. 
  - Make sure to add `"{INSTALL_LOCATION}/bin"` to the environment `PATH`.
- Run `python -m retro.scripts.playback_movie path_to_recording` where `path_to_recording` can be replaced with the path to an actual recording.