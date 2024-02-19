#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def compare_output(var_old, var_new, bil = True):
    if bil == True:
        base = pd.DataFrame(pd.Series(var_old.extract_values())).reset_index().rename(columns={'level_0':'from_iso3','level_1':'to_iso3',0:'base'})
        new = pd.DataFrame(pd.Series(var_new.extract_values())).reset_index().rename(columns={'level_0':'from_iso3','level_1':'to_iso3',0:'new'})
        compare = base.merge(new, on = ['from_iso3','to_iso3'])
    else:
        base = pd.DataFrame(pd.Series(var_old.extract_values())).reset_index().rename(columns={'index':'iso3',0:'base'})
        new = pd.DataFrame(pd.Series(var_new.extract_values())).reset_index().rename(columns={'index':'iso3',0:'new'})
        compare = base.merge(new, on = ['iso3'])

    compare['diff'] = compare['new'] - compare['base']
    compare['diff_perc'] = np.round(100*(compare['new'] - compare['base']) / compare['base'], 1).replace(np.inf, 100).replace(np.nan, 0)

    return compare

def round_dict(d, k):
    return {key: float(f"{value:.{k}f}") for key, value in d.items()}

def shock(prod_dict, factor):
    return {key: value * factor for key, value in prod_dict.items()}

def dataframe_model_output(var1, var2):
    predicted = pd.DataFrame(pd.Series(var1.extract_values())).reset_index().rename(columns={'level_0':'from_iso3','level_1':'to_iso3',0:'new'})
    base = pd.DataFrame(pd.Series(var2.extract_values())).reset_index().rename(columns={'level_0':'from_iso3','level_1':'to_iso3',0:'original'})

    validation = predicted.merge(base, on = ['from_iso3','to_iso3'])
    return validation


def remove_countries(country, bil_trade):
    #### only include countries that either demand crop or supply it ###
    country = country[(country['demand_q']>0) | (country['supply_q']>0)].reset_index(drop = True)

    bil_trade = bil_trade[(bil_trade['from_iso3'].isin(country['iso3'].unique())) &
                                   (bil_trade['to_iso3'].isin(country['iso3'].unique()))].reset_index(drop = True)

    return country,  bil_trade


class CountryClass:

    def __init__(self, dataframe, stocks):

        ### processing
        dataframe['share_fertilizer'] = dataframe['Fertilizer_USD_t']/dataframe['Production_USD_t']
        dataframe['share_pesticides'] = dataframe['Pesticides_USD_t']/dataframe['Production_USD_t']
        dataframe['share_diesel'] = dataframe['Diesel_USD_t']/dataframe['Production_USD_t']
        dataframe['share_labour'] = dataframe['Labour_USD_t']/dataframe['Production_USD_t']
        dataframe['share_machinery'] = dataframe['Machinery_USD_t']/dataframe['Production_USD_t']

        ### countries list
        self.iso3 = list(dataframe['iso3'])

        ### production cost origin country
        self.production_cost = dataframe.set_index(['iso3'])['Production_USD_t']

        ### production shares
        self.share_fertilizer = dataframe.set_index(['iso3'])['share_fertilizer']
        self.share_pesticides = dataframe.set_index(['iso3'])['share_pesticides']
        self.share_diesel = dataframe.set_index(['iso3'])['share_diesel']
        self.share_labour = dataframe.set_index(['iso3'])['share_labour']
        self.share_machinery = dataframe.set_index(['iso3'])['share_machinery']

        ## demand and supply elasticity
        self.demand_elas = dataframe.set_index(['iso3'])['demand_elas']*-1 ### make a positive number
        self.supply_elas = dataframe.set_index(['iso3'])['supply_elas']

        self.stocks = stocks.set_index(['iso3'])['stocks_model']
        self.stocks_min = stocks.set_index(['iso3'])['stocks_model_min']
        self.stocks_max = stocks.set_index(['iso3'])['stocks_model_max']
        self.stocksuse = stocks.set_index(['iso3'])['stocks-to-use']
        ### total demand and supply
        self.demand = dataframe.set_index(['iso3'])['demand_q']/1000
        self.supply = dataframe.set_index(['iso3'])['supply_q']/1000


class BilateralClass:

    def __init__(self, dataframe, factor_error):
        error = 1*10**(-1*factor_error)
        dataframe['q_calib'] = dataframe['q_calib']/1000
        dataframe['q_old'] = dataframe['q_old']/1000

        dataframe['q_calib'] = np.where(dataframe['q_calib']<=error, error, dataframe['q_calib'])
        dataframe['q_old'] = np.where(dataframe['q_old']<=error, error, dataframe['q_old'])

        ### adjust internal trade ###
        dataframe['q_old'] = np.where(((dataframe['from_iso3'] == dataframe['to_iso3'])&(dataframe['trade_relationship_old']==0)),
                                                        dataframe['q_calib'], dataframe['q_old'])

        dataframe['trade_relationship_old'] = np.where(((dataframe['from_iso3'] == dataframe['to_iso3'])&(dataframe['trade_relationship']==1)),
                                                        1, dataframe['trade_relationship_old'])

        ## calibration trade
        self.trade01 =  np.round(dataframe.set_index(['from_iso3','to_iso3'])['q_calib'], factor_error) ## thousand of tonnes

        ## existing trade
        self.trade_old = np.round(dataframe.set_index(['from_iso3','to_iso3'])['q_old'], factor_error) ## thousand of tonnes

        ### binary for existing trade relation
        self.trade_binary = dataframe.set_index(['from_iso3','to_iso3'])['trade_relationship_old']

        ### trade cost
        self.tc1 = dataframe.set_index(['from_iso3','to_iso3'])['trade_USD_t']

        ## ad-valorem tariff
        self.adv = dataframe.set_index(['from_iso3','to_iso3'])['adv']

def read_model_input(crop_code, error_factor):
    file_country = 'Processed/Country_data/country_information_'+str(crop_code)+'.csv'
    file_bil = 'Processed/Trade_cost/bilateral_trade_cost_'+str(crop_code)+'.csv'

    #### read data ###
    country_data = pd.read_csv(file_country)
    bil_trade_data = pd.read_csv(file_bil)

    ## stocks ###
    stocks = pd.read_csv('Input/Stocks/stocks_processed.csv')
    stocks_crop = stocks[stocks['crop_code']==crop_code].reset_index(drop = True)
    ### remove countries where demand and supply are zero
    country_data, bil_trade_data = remove_countries(country = country_data, bil_trade = bil_trade_data)

    stocks_crop = stocks_crop[stocks_crop['iso3'].isin(country_data['iso3'].unique())].reset_index(drop = True)
    ### create two classes
    country_class = CountryClass(country_data, stocks_crop)
    bilateral_class = BilateralClass(bil_trade_data, factor_error = error_factor)

    return country_class, bilateral_class


def compare_trade_flows(model_predict, model_base, error):
    ### predicted trade flows
    predicted_trade = pd.DataFrame(pd.Series(model_predict.extract_values())).reset_index().rename(columns={'level_0':'from_iso3','level_1':'to_iso3',0:'t_pred'})
    base_trade = pd.DataFrame(pd.Series(model_base.extract_values())).reset_index().rename(columns={'level_0':'from_iso3','level_1':'to_iso3',0:'t_exist'})

    ### merge
    trade_validation = predicted_trade.merge(base_trade, on = ['from_iso3','to_iso3'])
    trade_validation['t_pred'] = np.where(trade_validation['t_pred']<error, error, trade_validation['t_pred'])
    trade_validation['error'] = (trade_validation['t_pred']-trade_validation['t_exist'])

    ### binary predicton ###
    trade_validation['correct'] = np.where((trade_validation['t_pred']>1)&(trade_validation['t_exist']>1), 1, 0)

    ### create some statistics  ###
    hit_ratio = trade_validation['correct'].sum()/len(trade_validation[trade_validation['t_pred']>1])
    MSE = mean_squared_error(trade_validation['t_exist'].values,trade_validation['t_pred'].values)
    r2 = r2_score(trade_validation[trade_validation['from_iso3']!=trade_validation['to_iso3']]['t_exist'].values,trade_validation[trade_validation['from_iso3']!=trade_validation['to_iso3']]['t_pred'].values)

    print('hit ratio:', np.round(hit_ratio, 2))
    print('MSE:', np.round(MSE, 2))
    print('R-squared:',np.round(r2, 2))


    return trade_validation, hit_ratio, MSE, r2


def scatter_plot_trade(df_output):
    ### plot of trade flows ##
    fig, ax = plt.subplots(figsize = (3.5,3.5))
    plt.scatter(np.log(df_output['t_pred']+1), np.log(df_output['t_exist']+1), s = 8, color = 'grey', alpha = 0.4, zorder = 3)
    plt.plot(np.linspace(0,13, 100), np.linspace(0,13, 100), color = 'k', lw = 0.5, zorder = 1)
    plt.xlim(0, 12); plt.xlabel('Predicted trade ln (x1,000 t)');
    plt.ylim(0, 12); plt.ylabel('Observed trade ln (x1,000 t)');
