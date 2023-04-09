from quadrotor_agent import quadrotor_agent
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
    Mode1 = auto() # mode of the uncertain mass (from 0.5m to 3.5m)
    Mode2 = auto() # mode of the nominal mass
    Mode3 = auto() # deterministic mass and 3 times of the nominal value
    Mode4 = auto() # mode of the uncertain mass (from 0.5m to 3.5m), without L1
    Mode5 = auto() # mode of the uncertain mass (from 0.5m to 3.5m), with L1
    Mode6 = auto() # mode of bunch of mass parameters (after 6 seconds, L1 is switched on)
    Mode7 = auto() # mode of changing mass (one mass at one time, instead of 'set-wise' verification)
    Mode8 = auto() # new mode added to explore the effect of time delay (dde integrated)


if __name__ == "__main__":
    input_code_name = './quad_controller.py'
    scenario = Scenario()

    # step 1. create a quadrotor instance with the closed-loop dynamics
    quad = quadrotor_agent('quad1', file_name=input_code_name)
    scenario.add_agent(quad)
    # car = vanderpol_agent('car2', file_name=input_code_name)
    # scenario.add_agent(car)
    # scenario.set_sensor(FakeSensor2())
    # modify mode list input
    # Step 2. change the initial codnitions (set for all 18 states)
    scenario.set_init(
        [
             #20 states: (position, velocity, Rotation matrix, omega, mass, time)
        [   
            ### the two lines below are for the verification of transient performance (undram_change + drama_change + delay)
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 7.34, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 9.34, 0.0]
            ### the two lines below are for the verification of transient performance (dram_change_aggres)
            # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.34, 0.0],
            # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5.34, 0.0]

            # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.5, 0.0]
        ], # tuning knobs (initial condition uncertainty)
            # [[1.25], [1.25]],
            # [[1.25, 2.25], [1.25, 2.25]],
            # [[1.55, 2.35], [1.55, 2.35]]
        ],
        [
            tuple([AgentMode.Mode8]),
            # tuple([AgentMode.Default]),
        ]
    )
    t_max = 10 # for fast mass change only
    # t_max= 9.42
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
    traces.dump('./demo/quadrotor_l1ac/output_result_dm_L1AC_015.json') 
    # fig = go.Figure()
    # # """use these lines for generating x-y (phase) plots"""
    # fig = reachtube_tree(traces, None, fig, 0, 1,
    #                         'lines', 'trace', print_dim_list=[1,2])
    # fig.add_trace(go.Scatter(x=t, y=x_des_array,mode="lines",line=dict(color="#0000ff"))) 
    # fig.update_xaxes(range=[-0.5, 10.5],constrain="domain")
    # fig.show()
    # fig = go.Figure()
    # """use these lines for generating x-y (phase) plots"""
    # fig = simulation_tree(traces, None, fig, 0, 2,
    #                         'lines', 'trace', print_dim_list=[1,2])
    # fig.add_trace(go.Scatter(x=t, y=y_des_array,mode="lines",line=dict(color="#0000ff"))) 
    # fig.update_xaxes(range=[-0.5, 10.5],constrain="domain")
    # fig.show()
    # fig = go.Figure()
    # """use these lines for generating x-y (phase) plots"""
    # fig = simulation_tree(traces, None, fig, 1, 2,
    #                         'lines', 'trace', print_dim_list=[1,2])
    # fig.add_trace(go.Scatter(x=x_des_array, y=y_des_array,mode="lines",line=dict(color="#0000ff"))) 
    # fig.update_xaxes(range=[-0.5, 10.5],constrain="domain")
    # fig.show()

    # traces = scenario.verify(t_max, 0.05)
    # plt.rcParams['figure.dpi'] = 200
    # plt.rcParams['savefig.dpi'] = 200
    # fig = plt.figure(0)
    # plt.plot(t, x_des_array,'r--')
    # plt.xlabel('t [sec]')
    # plt.ylabel('x [m]')
    # fig = plot_reachtube_tree(traces.root, 'quad1', 0, [1], fig=fig)
    # # fig = plot_simulation_tree(traces.root, 'quad1', 0, [1], fig=fig)
    # ax = fig.gca()
    # ax.legend(['desired','actual'],fontsize="20")
    # # ax.set_aspect('equal', 'box')
    # plt.rcParams.update({'font.size': 20})
    # plt.savefig("delay_verification_L1_uncertain_xt.png")

    # plt.rcParams['figure.dpi'] = 200
    # plt.rcParams['savefig.dpi'] = 200
    # fig = plt.figure(1)
    # plt.plot(t, y_des_array,'r--',label='desired')
    # plt.xlabel('t [sec]')
    # plt.ylabel('y [m]')
    # fig = plot_reachtube_tree(traces.root, 'quad1', 0, [2], fig=fig)
    # # fig = plot_simulation_tree(traces.root, 'quad1', 0, [2], fig=fig)
    # ax = fig.gca()
    # ax.legend(['desired','actual'],fontsize="20")
    # plt.rcParams.update({'font.size': 20})
    # # ax.set_aspect('equal', 'box')
    # plt.savefig("delay_verification_L1_uncertain_yt.png")

    # plt.rcParams['figure.dpi'] = 200
    # plt.rcParams['savefig.dpi'] = 200
    
    
    # The new plotter: webpage-based visualization
#     fig = go.Figure()
#     fig = reachtube_tree(traces, None, fig, 0, 3,plot_color=[['#0000CC', '#0000FF', '#3333FF', '#6666FF', '#9999FF', '#CCCCFF']], name='actual')
    
#     fig.update_xaxes(title_text="t [sec]")
#     fig.update_yaxes(title_text="z [m]")
#     fig.add_trace(go.Scatter(
#     x=t,
#     y=z_des_array,
#     line_color = "red",      
#     name = 'desired'# this sets its legend entry
# ))
#     fig.update_layout(
#     title="Reach Tube",
#     font=dict(
#         family="Courier New, monospace",
#         size=18,
#         color="RebeccaPurple"
#     )
# )
#     fig.show()

    fig = plt.figure(3)
    plt.plot(t, z_des_array,'r--',label='desired')
    plt.xlabel('t [sec]')
    plt.ylabel('z [m]')
    fig = plot_reachtube_tree(traces.root, 'quad1', 0, [3], fig=fig)
    # fig = plot_simulation_tree(traces.root, 'quad1', 1, [2], fig=fig)
    ax = fig.gca()
    ax.legend(['desired','actual'])
    plt.rcParams.update({'font.size': 16})
    # ax.set_aspect('equal', 'box')
    plt.show()
    
    
    
    # fig = plot_simulation_tree(traces.root, 'quad1', 0, [3], fig=fig)
    # ax = fig.gca()
    # ax.annotate("$\mathcal{L}_1$AC switches on", xy=(6, -0.2), xytext=(5, -1), arrowprops={"arrowstyle":"->", "color":"gray"})
    # ax.legend(['desired','actual'],fontsize="20")
    # ax.tick_params(axis='x', labelsize=16)
    # ax.tick_params(axis='y', labelsize=16)
    # plt.rcParams.update({'font.size': 16})
    # ax.set_aspect('equal', 'box')
    # plt.savefig("delay_verification_L1_uncertain_zt.png")

    # ax.annotate("1.3$m_0$", xy=(2, 1.4), xytext=(1.5, 0.6), arrowprops={"arrowstyle":"->", "color":"gray"})
    # ax.annotate("$m_0$", xy=(4, 1.7),   ytext=(3.75, 1), arrowprops={"arrowstyle":"->", "color":"gray"})
    # ax.annotate("0.65$m_0$", xy=(6, 0.), xytext=(5.4, 0.7), arrowprops={"arrowstyle":"->", "color":"gray"})
    # plt.rcParams['figure.dpi'] = 200
    # plt.rcParams['savefig.dpi'] = 200
   
    # plt.savefig("delay_verification_L1_uncertain_xy.png")
    # plt.show()
    # path = os.path.abspath(__file__)
    # path = path.replace('quadrotor_demo.py', 'output_geo_L1_TEST.json')
    # write_json(traces, path)



    # for t_step in t:
    #     x_des_array.append(2*(1-np.cos(t_step)))
    #     y_des_array.append(2*np.sin(t_step))
    #     z_des_array.append(1.0 + np.sin(t_step))
    # fig = go.Figure()
    # """use these lines for generating x-y (phase) plots"""
    # fig = reachtube_tree(traces, None, fig, 1, 2,
    #                         'lines', 'trace', print_dim_list=[1,2])
    # fig.add_trace(go.Scatter(x=x_des_array, y=y_des_array,mode="lines",line=dict(color="#0000ff"))) 
    # fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
    # fig.update_xaxes(range=[-0.5, 4.5],constrain="domain")
    # fig.show()

    # """use these lines for generating x-t (time) plots"""
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 0, 1,
    #                       'lines', 'trace', print_dim_list=[0,1])
    # fig.add_trace(go.Scatter(x=t, y=x_des_array,mode="lines",line=dict(color="#0000ff"))) 
    # fig.update_xaxes(range=[-0.5, 10.5],constrain="domain")
    # fig.show()


    # fig_n = go.Figure()
    # fig_n = reachtube_tree(traces, None, fig_n, 0, 2,
    #                       'lines', 'trace', print_dim_list=[0,2])
    # fig_n.add_trace(go.Scatter(x=t, y=y_des_array,mode="lines",line=dict(color="#0000ff"))) 
    # fig_n.update_xaxes(range=[-0.5, 10.5],constrain="domain")
    # fig_n.show()

    # fig_new = go.Figure()
    # fig_new = reachtube_tree(traces, None, fig_new, 0, 3,
    #                       'lines', 'trace', print_dim_list=[0,3])
    # fig_new.add_trace(go.Scatter(x=t, y=z_des_array,mode="lines",line=dict(color="#0000ff"))) 
    # fig_new.update_xaxes(range=[-0.5, 10.5],constrain="domain")
    # fig_new.show()
