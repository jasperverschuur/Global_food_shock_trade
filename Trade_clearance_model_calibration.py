#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.mpec import *
import math
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from functions_general import *
from functions_calibration import *
from functions_shock import *
from pathos.multiprocessing import ProcessPool, cpu_count



def create_file_name(crop_code, shock_dict):
    ### create filename of crop with shocks added ###
    file_name = crop_code
    for item in shock_dict.items():
        if item[1] == 'yes':
            file_name = file_name+'_'+item[0]

    return file_name


###### additional information,  ####
sigma_val = 10 ### base = 10,
eps_val = 7 ### base = 7

factor_error = 3
error = 1*10**(-1*factor_error)
calibration_output = 'Output/Calibration/'

#### read data ###
for crop_code in ['RICE','MAIZ','SUGB','GROU','WHEA','SOYB','SORG','SUNF','POTA','BARL']:
    print(crop_code)
    file_country = 'Processed/Country_data/country_information_'+str(crop_code)+'.csv'
    file_bil = 'Processed/Trade_cost/bilateral_trade_cost_'+str(crop_code)+'.csv'

    ### read and process model input ###
    country_class, bilateral_class = read_model_input(crop_code, factor_error)


    ### solve the first model step ###
    print(datetime.datetime.now())
    model_step1, trade_calibration_1 = transport_cost_model(country_info = country_class,
                                                            bilateral_info = bilateral_class,
                                                            sigma_val = sigma_val,
                                                            eps_val = eps_val,
                                                            error = error,
                                                            linear = 'no')

    print(datetime.datetime.now())

    ### model validation after step 1 ###
    model_validation_s1, hit_ratio_s1, MSE_s1, R2_s1 = compare_trade_flows(model_predict = model_step1.trade1,
                                                               model_base = model_step1.trade01,
                                                               error = error)

    ### plot of trade flows ##
    scatter_plot_trade(df_output = model_validation_s1)


    #### run step 2 calibration ####
    model_calibration = trade_clearance_calibration(country_info = country_class,
                                                    bilateral_info = bilateral_class,
                                                    sigma_val = sigma_val,
                                                    eps_val = eps_val,
                                                    error = error,
                                                    trade_calibration_step1 = trade_calibration_1,
                                                    crop_code = crop_code,
                                                    output_file = calibration_output,
                                                    count_max = 30,  ### default = 25
                                                    mu_val = 0.01,  ### default = 0.01
                                                    wtc = 10,  ### default = 10
                                                    wp = 5,    ### default = 5
                                                    wx = 200,  ### default = 200
                                                    max_iter = 500)  ### default = 500


    ### model validation after step 3 ###
    model_validation_s2, hit_ratio_s2, MSE_s2, R2_s2 = compare_trade_flows(model_predict = model_calibration.trade2,
                                                               model_base = model_calibration.trade01,
                                                               error = error)
    ### plot of trade flows ##
    scatter_plot_trade(df_output = model_validation_s2)
