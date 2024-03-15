from typing import List, Tuple

import numpy as np
import torch
import verse
from numba import njit
from scipy.integrate import ode
# from tutorial_utils import drone_params
from verse import BaseAgent, LaneMap
from verse.analysis.analysis_tree import TraceType
from verse.analysis.utils import wrap_to_pi
from verse.map.lane_map_3d import LaneMap_3d
from tutorial_sensor import DefaultSensor
from dl_acas import CraftMode

from aircraft_agent import AircraftAgent
from tutorial_map import M4
from verse.scenario import Scenario, ScenarioConfig

import warnings

import pyvista as pv
from verse.plotter.plotter3D import *

import plotly.graph_objects as go
from verse.plotter.plotter2D import *

from verse import Scenario
from verse.plotter import *
import numpy as np 
import plotly.graph_objects as go
from enum import Enum, auto

import matplotlib.pyplot as plt
from verse.plotter.plotter2D import reachtube_tree, simulation_tree
from verse.plotter.plotter2D_old import plot_reachtube_tree, plot_simulation_tree
import os

import functools as ft

import ipdb
import jax
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import tqdm # for progress bar
from jax_guam.functional.guam_new import FuncGUAM, GuamState
from jax_guam.utils.jax_utils import jax2np, jax_use_cpu, jax_use_double
from jax_guam.utils.logging import set_logger_format
from loguru import logger

ego_mode = CraftMode.Coc
int_mode = CraftMode.Coc

decisions_ego = np.load("commands_671.npy")
decisions_ego = decisions_ego.tolist()

scenario = Scenario()
# scenario.set_map(M4())

ac1 = AircraftAgent(
    "aircraft1", file_name="dl_acas.py", initial_mode=ego_mode)

# set of the initial states:
# 1. make sure x[12] is identical for ac1 and ac2 (i.e., co-altitude);
# 2. make sure the quaternion (x[15:19]) is set as [1. 0, 0, 0] (if no pitch, roll, yaw)


ac1.set_initial(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0, 0.0, 0.0, 0.0, 10, 0, -10, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0, 0.0, 0.0, 0.0, 10, 0, -10, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.0]
    ],
    ([CraftMode.Coc]),
)

# decisions_int = [0] * 96
ac2 = AircraftAgent(
    "aircraft2", file_name="dl_acas.py", initial_mode=int_mode)

ac2.set_initial(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.8, 3.3, 0, 0.0, 0.0, 0.0, 0, 0, -10, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.8, 3.3, 0, 0.0, 0.0, 0.0, 0, 0, -10, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.0]
    ],
    ([CraftMode.Coc]),
)

scenario.add_agent(ac1)
scenario.add_agent(ac2)

scenario.set_sensor(DefaultSensor())

traces_simu = scenario.simulate(19.2, 0.01)
traces_veri = scenario.verify(19.2, 0.01)

warnings.filterwarnings("ignore")

fig = go.Figure()
fig = reachtube_tree(traces_veri, None, fig, 13, 14, plot_color= [['#0000CC', '#0000FF', '#3333FF', '#6666FF', '#9999FF', '#CCCCFF'], # blue
                                                                  ['#CC0000', '#FF0000', '#FF3333', '#FF6666', '#FF9999', '#FFCCCC']])  # red
fig.show()

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
# `fig4 = go.Figure()` is creating a new figure object using Plotly's graph_objects module. This
# `go.Figure()` function initializes a new figure that can be used to create interactive plots
# using Plotly. This figure object can then be used to add traces, annotations, layout settings,
# and more to create visualizations in Plotly.
# fig4 = go.Figure()

plt.xlabel('t [sec]')
plt.ylabel('x [m]')
fig1 = plot_reachtube_tree(traces_veri.root, 'aircraft2', 0, [13], color='r', fig=fig1)
fig1 = plot_reachtube_tree(traces_veri.root, 'aircraft1', 0, [13], color='b', fig=fig1)
fig1.show()

plt.xlabel('t [sec]')
plt.ylabel('y [m]')
fig2 = plot_reachtube_tree(traces_veri.root, 'aircraft2', 0, [14], color='r', fig=fig2)
fig2 = plot_reachtube_tree(traces_veri.root, 'aircraft1', 0, [14], color='b', fig=fig2)
fig2.show()

plt.xlabel('t [sec]')
plt.ylabel('z [m]')
fig3 = plot_reachtube_tree(traces_veri.root, 'aircraft2', 0, [15], color='r', fig=fig3)
fig3 = plot_reachtube_tree(traces_veri.root, 'aircraft1', 0, [15], color='b', fig=fig3)
fig3.show()


# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# fig4 = reachtube_tree(traces_veri, None, fig4, 13, 14,
#                             print_dim_list=[1,2])
# fig4.show()


plt.tight_layout()
plt.show()


