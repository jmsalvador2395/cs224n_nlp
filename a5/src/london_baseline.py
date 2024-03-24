# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import random
import argparse
random.seed(0)

import dataset
import model
import trainer
import utils

num_samples = len(open('birth_dev.tsv').read().split('\n')) - 1
total, correct = utils.evaluate_places('birth_dev.tsv', ['London']*num_samples)
print(f'{int(correct)} correct out of {int(total)} - {correct/total:.2%}')