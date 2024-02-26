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

class AgentMode(Enum):
    Mode1 = auto() 
    

if __name__ == "__main__":
    input_code_name = './guam_controller.py'
    scenario = Scenario()

    guam = guam_agent('guam1', file_name=input_code_name)
    scenario.add_agent(guam)
    scenario.set_init(
        [
            # TODO: Fix the following upper and lower bounds of the states' initial conditions
             #20 states: (position, velocity, Rotation matrix, omega, mass, time)
        [   
            ### the two lines below are for the verification of transient performance (undram_change + drama_change + delay)
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 7.34, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 9.34, 0.0]
        ], 
        ],
        [
            tuple([AgentMode.Mode1]),
        ]
    )
    t_max = 45 # for fast mass change only

    # TODO: plot the reference input for the Guam model
    N = int(100*t_max + 1)
    t = np.linspace(0,t_max,N)
    x_des_array = []
    y_des_array = []
    z_des_array = []

    for t_step in t:
        x_des_array.append(2*(1-np.cos(t_step)))
        y_des_array.append(2*np.sin(t_step))
        z_des_array.append(1-np.cos(t_step))

    traces = scenario.verify(t_max, 0.01)
    traces.dump('./demo/guam/output_result_guam.json') 

    """TODO (minor): fix the following plotter"""
    # fig = plt.figure(3)
    # plt.plot(t, z_des_array,'r--',label='desired')
    # plt.xlabel('t [sec]')
    # plt.ylabel('z [m]')
    # fig = plot_reachtube_tree(traces.root, 'quad1', 0, [3], fig=fig)
    # # fig = plot_simulation_tree(traces.root, 'quad1', 1, [2], fig=fig)
    # ax = fig.gca()
    # ax.legend(['desired','actual'])
    # plt.rcParams.update({'font.size': 16})
    # # ax.set_aspect('equal', 'box')
    # plt.show()
    
    
    
