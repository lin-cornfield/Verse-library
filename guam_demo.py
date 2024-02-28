from guam_agent import guam_agent
from verse import Scenario
from verse.plotter import *
import numpy as np 
import plotly.graph_objects as go
from enum import Enum, auto

import matplotlib.pyplot as plt
from verse.plotter.plotter2D import reachtube_tree
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
from jax_guam.subsystems.genctrl_inputs.genctrl_inputs import lift_cruise_reference_inputs
from jax_guam.utils.jax_utils import jax2np, jax_use_cpu, jax_use_double
from jax_guam.utils.logging import set_logger_format
from loguru import logger
class AgentMode(Enum):
    Mode1 = auto() 
    
""" some comments in running the verification:
1. The verification step size is set as the guam sim step size (in the guam agent initialization) for simplicity;
2. Sensitive to the initial condition choice"""

if __name__ == "__main__":
    input_code_name = './guam_controller.py'
    scenario = Scenario()

    guam = guam_agent('guam1', file_name=input_code_name)
    scenario.add_agent(guam)
    scenario.set_init(
        [
            # TODO: Fix the following upper and lower bounds of the states' initial conditions
             #24 states (6 Control States, 13 Aircraft States, 5 Surf Eng state)
        [   
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.00069017, 0, -8, 0.0, 0.0, 0.0, -0.01, -0.01, 0.0, 1.0, 0.0, -4.3136e-05, 0.0, 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.00069017, 0, -8, 0.0, 0.0, 0.0, 0.01, 0.01, 0.0, 1.0, 0.0, -4.3136e-05, 0.0, 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0]
        ], 
        ],
        [
            tuple([AgentMode.Mode1]),
        ]
    )
    t_max = 10 # for fast mass change only

    # TODO: plot the reference input for the Guam model
    N = int(100*t_max + 1)
    t = np.linspace(0,t_max,N)
    
    x_des_array = []
    y_des_array = []
    z_des_array = []

    for t_step in t:
        ref_inputs = lift_cruise_reference_inputs(t_step)
        Pos_des = ref_inputs.Pos_des
        x_des_array.append(Pos_des[0])
        y_des_array.append(Pos_des[1])
        z_des_array.append(Pos_des[2])

    traces = scenario.verify(t_max, 0.01, params={"bloating_method": "GLOBAL"})
    # print(traces.root)
    # traces.dump('./demo/guam/output_result_guam.json') 

    """TODO (minor): fix the following plotter"""
    # fig = plt.figure(3)
    # # plt.plot(t, z_des_array,'r--',label='desired')
    # plt.xlabel('t [sec]')
    # plt.ylabel('x [m]')
    # fig = plot_reachtube_tree(traces.root, 'guam1', 0, [12], fig=fig)
    # # fig = plot_simulation_tree(traces.root, 'quad1', 1, [2], fig=fig)
    # ax = fig.gca()
    # # ax.legend(['desired','actual'])
    # plt.rcParams.update({'font.size': 16})
    # # ax.set_aspect('equal', 'box')
    # plt.show()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    # `fig4 = go.Figure()` is creating a new figure object using Plotly's graph_objects module. This
    # `go.Figure()` function initializes a new figure that can be used to create interactive plots
    # using Plotly. This figure object can then be used to add traces, annotations, layout settings,
    # and more to create visualizations in Plotly.
    fig4 = go.Figure()
    
    plt.xlabel('t [sec]')
    plt.ylabel('x [m]')
    fig1 = plot_reachtube_tree(traces.root, 'guam1', 0, [12], fig=fig1)
    
    plt.xlabel('t [sec]')
    plt.ylabel('y [m]')
    fig2 = plot_reachtube_tree(traces.root, 'guam1', 0, [13], fig=fig2)
    
    plt.xlabel('t [sec]')
    plt.ylabel('z [m]')
    fig3 = plot_reachtube_tree(traces.root, 'guam1', 0, [14], fig=fig3)
    
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    fig4 = reachtube_tree(traces, None, fig4, 12, 13,
                             print_dim_list=[1,2])
    fig4.show()
    
    
    plt.tight_layout()
    plt.show()
