import os

os.system("conda env create -f env.yml")
os.system("conda activate uu_ml2")
os.system("python -m retro.import ROMs")
