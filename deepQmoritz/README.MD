# UU_ML_Project
Uppsala University - ML - Project

## Notes
Please be aware that this implementation currently still uses `Tensorflow 1.3` and not `2.0`. The goal is to convert this in a future version!

## Install
- `python ./install/setup.py` (installation including conda environment)
  - If it doesn't work, `python -m pip install -r ./install/requirements.txt` should work. 
- Try if it worked by running:
    - `python -m retro.examples.interactive --game SuperMarioBros3-Nes`

## Rewards
Have a look at the files: `data.json` and `scenario.json` to see how the rewards are set. 

## Hyperparameters
The different hyperparameters and configurations can be found in: `config.py`