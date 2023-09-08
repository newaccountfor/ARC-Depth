from __future__ import absolute_import, division, print_function
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
from options_teacher import MonodepthOptions

from trainer_teacher import Trainer
options = MonodepthOptions()
opts = options.parse()

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
