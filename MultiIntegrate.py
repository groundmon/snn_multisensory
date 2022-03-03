#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 16:27:23 2022

@author: jyhjonghsieh
"""

from brian2 import *
import numpy as np
import os
import random
import math
import warnings

##### parameters
n_size = 6000
V_L = -70*mV
V_E = 0*mV
V_I = -80*mV

EPSPtoW=100.

##### parameters of lognormal distribution
sigma_l = 1.0
mu_l = log(0.2)+sigma_l*sigma_l

class ModulesNetwork:
    def __init__(self, E_rate, random_seed, lambda1_list, lambda2_list, period_list, folder_name):
        seed(random_seed)

        n_e = int(n_size*E_rate)
        n_i = n_size - n_e

        #### a leaky integrate-and-fire model
        eqs = '''
                dv/dt  = -ge/ms*(v-V_E)-gi/ms*(v-V_I)-(v-V_L)/(tau_mem*ms)  :volt
                dge/dt = -ge/(2*ms)                : 1
                dgi/dt = -gi/(2*ms)               : 1
                tau_mem: 1
                '''

        #### Neuroal population 1 & 2 are input modules.
        P1 = NeuronGroup(int(n_e+n_i), eqs, method='euler', threshold='v>-50*mV', reset='v=-60*mV', refractory=1*ms)
        P1.v= -70*mV
        P1e = P1[:int(n_e)]
        P1i = P1[int(n_e):]
        P1e.tau_mem=20
        P1i.tau_mem=10

        P2 = NeuronGroup(int(n_e+n_i), eqs, method='euler', threshold='v>-50*mV', reset='v=-60*mV', refractory=1*ms)
        P2.v= -70*mV
        P2e = P2[:int(n_e)]
        P2i = P2[int(n_e):]
        P2e.tau_mem=20
        P2i.tau_mem=10

        #### Population 3 is an integrate module.
        P3 = NeuronGroup(int(n_e+n_i), eqs, method='euler', threshold='v>-50*mV', reset='v=-60*mV', refractory=1*ms)
        P3.v= -70*mV
        P3e = P3[:int(n_e)]
        P3i = P3[int(n_e):]
        P3e.tau_mem=20
        P3i.tau_mem=10

        print("creating network...")
        ############# intramodule E2E  #############
        ### A function for EPSP synapse connection
        def EPSP_connect(P_i, P_j):
            C = Synapses(P_i, P_j, model="""w:1
                                            p:1""",
                                            on_pre='ge_post+=w*(rand()<p)')

            ## lognormal distribution of EPSP
            logN_EPSP = np.random.lognormal(mu_l, sigma_l, int(n_e*n_e/10))
            for i in range(0,len(logN_EPSP)):
                while logN_EPSP[i]>14:  #### amplitude larger than 14mV was considered as unrealistic value
                  logN_EPSP[i] = np.random.lognormal(mu_l, sigma_l)

            ## The coupling probability for EE connections is 0.1
            ## randomly pick up 10% from the all possible connection pairs (including self connections)
            num_ee = int(n_e*n_e*0.1)
            tmp0 = list(range(0, n_e*n_e))
            tmp = random.sample(tmp0, num_ee)
            ee_sorce = [0]*num_ee
            ee_target = [0]*num_ee

            #### ramdonly assign a source and a target from the neuron pair
            for i in range(0, len(tmp)):
                if random.random()>0.5:
                    ee_sorce[i] = math.floor(tmp[i]/n_e)
                    ee_target[i] = tmp[i]%n_e
                else:
                    ee_sorce[i] = tmp[i]%n_e
                    ee_target[i] = math.floor(tmp[i]/n_e)

            C.connect(i=ee_sorce, j=ee_target);
            C.w = logN_EPSP[:int(n_e*n_e/10)]/EPSPtoW
            C.delay = '(1+2*rand())*ms'
            C.p = 1.-0.1/(0.1+EPSPtoW*C.w)

            return C

        Ce1e1 = EPSP_connect(P1e, P1e)
        Ce2e2 = EPSP_connect(P2e, P2e)
        Ce3e3 = EPSP_connect(P3e, P3e)

        #############  A function for synapse connection  #############
        def synapse_connect(P_i, P_j, trans_p, weight, delay_mid):
            C = Synapses(P_i, P_j, model="""w:1
                                          p:1""",
                                          on_pre='ge_post+=w')
            C.connect(p=trans_p)
            C.w = weight
            if delay_mid==1:
                C.delay = '(2*rand())*ms'
            elif delay_mid==2:
                C.delay = '(1+2*rand())*ms'
            else:
                C.delay = '(2*rand())*ms'
                warnings.warn('The synaptic delay is not well defined, it was set as a default range (0-2) currently.')
            C.p = 1
            return C

        ### intermodule E2E
        Ce1e2 = synapse_connect(P1e, P2e, trans_p=0.01, weight=0.05, delay_mid=2)
        Ce2e1 = synapse_connect(P2e, P1e, trans_p=0.01, weight=0.05, delay_mid=2)
        Ce3e1 = synapse_connect(P3e, P1e, trans_p=0.01, weight=0.05, delay_mid=2)
        Ce3e2 = synapse_connect(P3e, P2e, trans_p=0.01, weight=0.05, delay_mid=2)

        ## log-normal distribution connect from input to integrate module
        Ce1e3 = EPSP_connect(P1e, P3e)
        Ce2e3 = EPSP_connect(P2e, P3e)

        ### intramodule E2I
        Ce1i1 = synapse_connect(P1e, P1i, trans_p=0.1, weight=0.018, delay_mid=1)
        Ce2i2 = synapse_connect(P2e, P2i, trans_p=0.1, weight=0.018, delay_mid=1)
        Ce3i3 = synapse_connect(P3e, P3i, trans_p=0.1, weight=0.018, delay_mid=1)

        # ### intermodule E2I
        Ce1i2 = synapse_connect(P1e, P2i, trans_p=0.01, weight=0.021, delay_mid=1)
        Ce2i1 = synapse_connect(P2e, P1i, trans_p=0.01, weight=0.021, delay_mid=1)
        Ce1i3 = synapse_connect(P1e, P3i, trans_p=0.01, weight=0.021, delay_mid=1)
        Ce2i3 = synapse_connect(P2e, P3i, trans_p=0.01, weight=0.021, delay_mid=1)
        Ce3i1 = synapse_connect(P3e, P1i, trans_p=0.01, weight=0.021, delay_mid=1)
        Ce3i2 = synapse_connect(P3e, P2i, trans_p=0.01, weight=0.021, delay_mid=1)

        ### I2I (only intra)
        Ci1i1 = synapse_connect(P1e, P1i, trans_p=0.5, weight=0.025, delay_mid=1)
        Ci2i2 = synapse_connect(P2e, P2i, trans_p=0.5, weight=0.025, delay_mid=1)
        Ci3i3 = synapse_connect(P3e, P3i, trans_p=0.5, weight=0.025, delay_mid=1)

        ### I2E (only intra)
        Ci1e1 = synapse_connect(P1i, P1e, trans_p=0.5, weight=0.002, delay_mid=1)
        Ci2e2 = synapse_connect(P2i, P2e, trans_p=0.5, weight=0.002, delay_mid=1)
        Ci3e3 = synapse_connect(P3i, P3e, trans_p=0.5, weight=0.002, delay_mid=1)

        #############  input to the network  #############
        #### Input signal of poisson spike train

        # #corresponding 40Hz  1/(25)*1000
        # stimulus = TimedArray(np.tile([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 10000)*Hz, dt=1.*ms)
        # P_ex1 = PoissonGroup((n_e+n_i), rates='Lambda1*stimulus(t)')
        # P_ex2 = PoissonGroup((n_e+n_i), rates='Lambda2*stimulus(t)')

        P_ex1 = PoissonGroup((n_e+n_i), rates='Lambda1')
        P_ex2 = PoissonGroup((n_e+n_i), rates='Lambda2')

        #### input to module 1
        S1 = Synapses(P_ex1, P1, model="""w:1
                                      p:1""",
                                      on_pre='ge_post+=w')
        S1.connect(j='i')
        S1.w = 0.2

        #### input to module 2
        S2 = Synapses(P_ex2, P2, model="""w:1
                                      p:1""",
                                      on_pre='ge_post+=w')
        S2.connect(j='i')
        S2.w = 0.2


        #### monitors setting
        M1 = SpikeMonitor(P1)
        M2 = SpikeMonitor(P2)
        M3 = SpikeMonitor(P3)

        for n in range(len(period_list)):
            Lambda1 = lambda1_list[n]*Hz
            Lambda2 = lambda2_list[n]*Hz
            ## for timed array inputs
            # Lambda1 = lambda1
            # Lambda2 = lambda2
            run(period_list[n]*second, 'stdout')

        plt.plot(M1.t/ms, M1.i, '.')
        plt.show()

        plt.plot(M3.t/ms, M3.i, '.')
        plt.show()

        os.makedirs(folder_name)
        def result_output(M, P_name):
            data_i = np.zeros(len(M.i))
            data_t = np.zeros(len(M.i))
            for n in range(len(M.i)):
                data_i[n] = M.i[n]
                data_t[n] = M.t[n]*1000
            np.save(folder_name+'/raster'+P_name, data_i)
            np.save(folder_name+'/raster_t'+P_name, data_t)

        result_output(M1, '1')
        result_output(M2, '2')
        result_output(M3, 'integrate')


    # def saveState(self, folder_name):
    #     plt.plot()
    #     os.makedirs(folder_name)
    #     def result_output(M, P_name):
    #         data_i = np.zeros(len(M.i))
    #         data_t = np.zeros(len(M.i))
    #         for n in range(len(M.i)):
    #             data_i[n] = M.i[n]
    #             data_t[n] = M.t[n]*1000
    #         np.save(folder_name+'/raster'+P_name, data_i)
    #         np.save(folder_name+'/raster_t'+P_name, data_t)
    #
    #     result_output(self.M1, '1')
    #     result_output(self.M2, '2')
    #     result_output(self.M3, 'integrate')
