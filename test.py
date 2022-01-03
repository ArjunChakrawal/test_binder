# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 21:33:31 2022

@author: Arjun
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import optimize
import warnings

warnings.filterwarnings("ignore")

# %%

plt.style.use("ggplot")

# %%
if not os.path.exists("fig"):
    os.makedirs("fig/png")
    os.makedirs("fig/svg")
    os.makedirs("fig/tif")
    os.makedirs("fig/html")

print("Hello")