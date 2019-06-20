from smo import SMO
import numpy as np
from math import *


from flask import Flask
from flask import render_template
    from flask import json



# function that receives a numpy vector as input
def f1(v):
    return np.sum(v**2) + 1;

def f2(v):
    return -cos(v[0]) * cos(v[1]) * exp(-pow((v[0] - pi), 2) - pow((v[1] - pi), 2))

# lim_min = np.array([50, 50])
# lim_max = np.array([100, 100])

lim_min = np.array([-1, -1])
lim_max = np.array([2, 2])

pop_size = 10
dim = 2
global_lider_limit = 5
local_lider_limit = 10
n = 100

optimizer = SMO(pop_size, dim, global_lider_limit, local_lider_limit, n)

optimizer.set_function(f2, lim_min, lim_max)

optimizer.initialize()

optimizer.optimize()

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('grafico.html')
