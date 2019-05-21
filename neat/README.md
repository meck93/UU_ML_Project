# UU_ML_Project - NEAT branch
Uppsala University - ML - Project


## Uses retro, no longer uses gym_super_mario_bros module 
Found a way to get the x-position for  Super Mario Brothers 3 -NES, see set-up. 

## Set-up
- in the ROMs folder there are 2 files now: data.json and scenario. json:
    - data.json now has the x-position and x-position multiplier (x-pos, resets fr 255 to 0 when moving far enought o the right)
    - scenario.json now rewards improved x-position, not increased score
    You can safely replace the data.json you have in the folder that looks something like: ~/<ENVIRONMENT>/lib/python3.7/site-packages/retro/data/stable/SuperMarioBros3-Nes
I don't know how the new scenario file affects RL.

###OLD instructions
-  python setup.py
- Try if it worked by running:
    - python -m retro.examples.interactive --game SuperMarioBros3-Nes

## Navigation
### Interactive Mode 
In the interactive mode, the user can navigate Mario through the levels.
- Use arrows left/right to move left/right
- Use keys X/Y/Z (depending on keyboard layout) to jump

