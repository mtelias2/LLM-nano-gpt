# coding up a transformer


import numpy as np
import torch
import torch.nn as nn
import torch.optim as adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
from sklearn.model_selection import train_test_split
import os
import urllib.request
