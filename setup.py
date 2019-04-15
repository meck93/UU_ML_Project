import os

os.system("conda env create -f environment.yml")
os.system("conda activate uu_ml_project")
os.system("python -m retro.import ROMs")