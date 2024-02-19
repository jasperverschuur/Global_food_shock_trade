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

factor_error = 3 ### base = 3
error = 1*10**(-1*factor_error)
error_scale = 10 ### base = 10
calibration_output = 'Output/Calibration/'

#### run models ###
crop_code = 'RICE' ### SOYB, MAIZ, WHEA, RICE

### settings# ###
list_year = list(np.linspace(1961,2014, 54).astype(int).astype(str))
N_iteration = 3000

### number of pools
pool = ProcessPool(nodes=cpu_count()-4)

### set scenarios ###
yield_var_list = ['yes','yes','yes','yes','yes']
ukraine_list =   ['no','yes','no','no','yes']
price_list =     ['no','no','yes','no','yes']
export_list =    ['no','no','no','yes','yes']

### loop over list ###
for yield_var_bin, ukraine_bin, price_bin, export_bin in zip(yield_var_list, ukraine_list, price_list, export_list):
    ### read country data ###
    country_class, bilateral_class = read_model_input(crop_code, factor_error)

    ### dict with shocks ####
    shock_dict = {'yieldvar':yield_var_bin,
                  'locust': 'no',
                  'ukraine': ukraine_bin,
                  'price':price_bin,
                  'exportres':export_bin}
    file_name = create_file_name(crop_code, shock_dict)
    print(file_name)

    ### run in parallel
    output = pool.map(shock_trade_clearance,
                                    [country_class]*len(list_year),
                                    [bilateral_class]*len(list_year),
                                    [eps_val]*len(list_year),
                                    [sigma_val]*len(list_year),
                                    [crop_code]*len(list_year),
                                    [calibration_output]*len(list_year),
                                    [shock_dict]*len(list_year),
                                    list_year,
                                    [error]*len(list_year), ### error
                                    [error_scale]*len(list_year), ### default = 10
                                    [N_iteration]*len(list_year)) ###

    ### output ###
    country_output = pd.concat([output[i][1] for i in range(0,len(output))])
    trade = pd.concat([output[i][0] for i in range(0,len(output))])

    ## output
    country_output.to_csv('Output/Trade_allocation/Country_output/country_output_'+file_name+'.csv', index = False)
    trade.to_csv('Output/Trade_allocation/Trade_output/trade_output_'+file_name+'.csv', index = False)

    print(datetime.datetime.now())
