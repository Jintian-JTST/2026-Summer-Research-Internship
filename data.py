import uproot
import numpy as np
import matplotlib.pyplot as plt

file = uproot.open("run6A.root")
tree = file["et_spectrum"]

data = tree["energy"].array()
time = tree["time"].array()

threshold = 0.6
mask = data > threshold
plt.hist(time[mask], bins=100)