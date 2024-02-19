#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.mpec import *
import math
import datetime
from functions_general import *


def locust_outbreak(country_dict):
    #### 2021 locust outbreak in Horn of Africa, Arabia Peninsula and South Asia
    #### and the 2023 locust outbreak in Afghanistan together ##
    for key, value in country_dict.supply.items():
        if key == 'PAK':
            country_dict.supply[key] = value * (0.82)
        elif key == 'SSD':
            country_dict.supply[key] = value * (0.89)
        elif key == 'ETH':
            country_dict.supply[key] = value * (0.81)
        elif key == 'SOM':
            country_dict.supply[key] = value * (0.95)
        elif key == 'UGA':
            country_dict.supply[key] = value * (0.96)
        elif key == 'KEN':
            country_dict.supply[key] = value * (0.96)
        elif key == 'IRN':
            country_dict.supply[key] = value * (0.97)
        elif key in ['AFG']:
            country_dict.supply[key] = value * (0.75)
        else:
            country_dict.supply[key] = value

    return country_dict

def input_price_spike(country_dict, prodprice_dict, p_fer = 1, p_pes = 1, p_die = 1, p_lab = 1, p_mach = 1):
    ### Model any kind of input price fluctuation ####
    #### input ##
    for key, value in country_dict.share_fertilizer.items():
        if value>0: ### if share values exist
            prodprice_dict[key] =  prodprice_dict[key]*((p_fer-1) * country_dict.share_fertilizer[key] +
                                                                                   (p_pes-1) * country_dict.share_pesticides[key] +
                                                                                   (p_die-1) * country_dict.share_diesel[key] +
                                                                                   (p_lab-1) * country_dict.share_labour[key] +
                                                                                   (p_mach-1) * country_dict.share_machinery[key])
        else: #### take the mean across countries otherwise
            prodprice_dict[key] = prodprice_dict[key]*((p_fer-1) * country_dict.share_fertilizer.mean() +
                                                                                   (p_pes-1) * country_dict.share_pesticides.mean() +
                                                                                   (p_die-1) * country_dict.share_diesel.mean() +
                                                                                   (p_lab-1) * country_dict.share_labour.mean() +
                                                                                   (p_mach-1) * country_dict.share_machinery.mean())


    return prodprice_dict


def ukraine_war(country_dict, bilateral_dict, tc_dict):
    ##### EU 28 countries ###
    EU_28 = ['AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA',
             'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD',
             'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE','GBR']

    ### Russian trade costs up ####
    for OD, value in tc_dict.items():
        if 'RUS' in OD:
            tc_dict[OD] = value * 1.5
        else:
            None

    #### Ukraine supply  ##
    for key, value in country_dict.supply.items():
        if key in ['UKR']:
            country_dict.supply[key] = value * 0.6
        else:
             None

    ### Trade restriction UKR and RUS ###
    for OD, value in tc_dict.items():
        if OD[0] == 'UKR' and OD[1] == 'RUS':
            tc_dict[OD] = value * 10
        elif OD[1] == 'UKR' and OD[0] == 'RUS':
            tc_dict[OD] = value * 10
        else:
            None

    ### Export restriction UKR and outside EU_28 ###
    for OD, value in tc_dict.items():
        if OD[0] == 'UKR' and OD[1] not in EU_28:
            tc_dict[OD] = value * 3
        elif OD[1] == 'UKR' and OD[0] not in EU_28:
            tc_dict[OD] = value * 3
        else:
            None

    ###
    return country_dict, bilateral_dict, tc_dict

def export_restrictions(tc_dict, trade_restriction_df):
    ### trade restrictions ###
    trade_restrictions_crop_nonan = trade_restriction_df[trade_restriction_df['from_iso3'].notna() & trade_restriction_df['to_iso3'].notna()].reset_index(drop = True)
    trade_restrictions_crop_fromna = trade_restriction_df[trade_restriction_df['from_iso3'].isna() & trade_restriction_df['to_iso3'].notna()].reset_index(drop = True)
    trade_restrictions_crop_tona = trade_restriction_df[trade_restriction_df['from_iso3'].notna() & trade_restriction_df['to_iso3'].isna()].reset_index(drop = True)

    ### implement  ###

    ### bilateral ###
    if len(trade_restrictions_crop_nonan)>0:
        for i in range(0, len(trade_restrictions_crop_nonan)):
            try:
                tc_dict[trade_restrictions_crop_nonan['from_iso3'].iloc[i],trade_restrictions_crop_nonan['to_iso3'].iloc[i]] = 10 * tc_dict[trade_restrictions_crop_nonan['from_iso3'].iloc[i],trade_restrictions_crop_nonan['to_iso3'].iloc[i]]
            except:
                continue

    ### import and export ban across all countries ###
    for key, value in tc_dict.items():
        if key[1] in list(trade_restrictions_crop_fromna['to_iso3'].unique()):
            tc_dict[key] = 10 * value

        if key[0] in list(trade_restrictions_crop_tona['from_iso3'].unique()):
            tc_dict[key] = 10 * value

    return tc_dict



def yield_variability(country_dict, yield_df, year):
    #### yield variability
    ### extract shock a make dict ###
    shock_dict = yield_df[['iso3',year]].set_index('iso3').squeeze().to_dict()

    #### Supply adjustment  ##
    for key, value in country_dict.supply.items():
        ### check
        try:
            country_dict.supply[key] = value * shock_dict[key]
        except:
            country_dict.supply[key] = value

    return country_dict

#### SHOCK MODEL ######
def process_final_output(model, year_select, crop_code, error, error_scale):
    ### extract data
    trade = pd.DataFrame(pd.Series(model.trade3.extract_values())).reset_index().rename(columns={'level_0':'from_iso3','level_1':'to_iso3',0:'trade'})

    ### adjust
    trade['trade'] = np.where(trade['trade']<= error/error_scale, 0, trade['trade'])
    trade['year'] = year_select
    trade['SPAM_code'] = crop_code

    ### supply
    supply = trade.groupby(['from_iso3'])['trade'].sum().reset_index().rename(columns = {'from_iso3':'iso3','trade':'supply'}).set_index(['iso3'])
    demand = trade.groupby(['to_iso3'])['trade'].sum().reset_index().rename(columns = {'to_iso3':'iso3','trade':'demand'}).set_index(['iso3'])

    ## domestic supply
    domestic_supply =  trade[trade['from_iso3']==trade['to_iso3']].groupby(['from_iso3'])['trade'].sum().reset_index().rename(columns = {'from_iso3':'iso3','trade':'dom_supply'}).set_index(['iso3'])
    import_supply =  trade[trade['from_iso3']!=trade['to_iso3']].groupby(['to_iso3'])['trade'].sum().reset_index().rename(columns = {'to_iso3':'iso3','trade':'import'}).set_index(['iso3'])
    export_supply =  trade[trade['from_iso3']!=trade['to_iso3']].groupby(['from_iso3'])['trade'].sum().reset_index().rename(columns = {'from_iso3':'iso3','trade':'export'}).set_index(['iso3'])

    ## prodprice, conprice
    prodprice = pd.DataFrame(pd.Series(model.prodprice3.extract_values())).reset_index().rename(columns={'index':'iso3',0:'prodprice'}).set_index(['iso3'])
    conprice = pd.DataFrame(pd.Series(model.conprice3.extract_values())).reset_index().rename(columns={'index':'iso3',0:'conprice'}).set_index(['iso3'])

    ### demand /supply output ###
    demand_output = pd.DataFrame(pd.Series(model.demand.extract_values())).reset_index().rename(columns={'index':'iso3',0:'demand_output'}).set_index(['iso3'])
    supply_output = pd.DataFrame(pd.Series(model.supply.extract_values())).reset_index().rename(columns={'index':'iso3',0:'supply_output'}).set_index(['iso3'])

    ### producer and consumer surplus ###
    B_value = pd.DataFrame(pd.Series(model.B.extract_values())).reset_index().rename(columns={'index':'iso3',0:'B_value'}).set_index(['iso3'])
    D_value = pd.DataFrame(pd.Series(model.D.extract_values())).reset_index().rename(columns={'index':'iso3',0:'D_value'}).set_index(['iso3'])


    ### merge together
    country_output = pd.concat([supply, demand, domestic_supply, import_supply, export_supply, prodprice, conprice, B_value, D_value, demand_output, supply_output], axis = 1)
    country_output['con_surplus_mUSD'] = (country_output['B_value']*country_output['demand']**2)/(2*1e6)
    country_output['prod_surplus_mUSD'] = (country_output['D_value']*country_output['supply']**2)/(2*1e6)

    country_output['year'] = year_select
    country_output['SPAM_code'] = crop_code
    return trade, country_output.reset_index()


def read_calibration_output(calibration_output_path, crop_code):
    #### read calibration files as dictionaries
    trade = pd.read_csv(calibration_output_path+'trade_calibration_'+crop_code+'.csv', header=None, index_col=[0,1]).squeeze().to_dict()
    prodprice = pd.read_csv(calibration_output_path+'prodprice_calibration_'+crop_code+'.csv', header=None, index_col=[0]).squeeze().to_dict()
    conprice = pd.read_csv(calibration_output_path+'conprice_calibration_'+crop_code+'.csv', header=None, index_col=[0]).squeeze().to_dict()
    tc = pd.read_csv(calibration_output_path+'tc_calibration_'+crop_code+'.csv', header=None, index_col=[0,1]).squeeze().to_dict()
    calib_constant = pd.read_csv(calibration_output_path+'calib_calibration_'+crop_code+'.csv', header=None, index_col=[0,1]).squeeze().to_dict()

    return trade, prodprice, conprice, tc, calib_constant

def price_increment_zero(increment_dict):
    new_dict = {}
    for key, value in increment_dict.items():
        new_dict[key] = 0
    return new_dict


def implement_shock_scenario(shock_dict, country_info, bilateral_info, trade_calib, prodprice_calib, conprice_calib, tc_calib, calib_constant, trade_restrictions_df, yield_var_df, year_select):
    prodprice_increment = prodprice_calib.copy()
    #### shocks ####

    ### LOCUST outbreak ###
    if shock_dict['locust'] == 'yes':
        country_info = locust_outbreak(country_dict = country_info)

    ### Price spike outbreak ###
    if shock_dict['price']=='yes':
        prodprice_increment = input_price_spike(country_dict = country_info, prodprice_dict = prodprice_increment,  p_fer = 3, p_pes = 3, p_die = 2, p_lab = 1, p_mach = 1)
    else:
        prodprice_increment = price_increment_zero(increment_dict = prodprice_increment)

    ### UKRAINE war ###
    if shock_dict['ukraine']=='yes':
        country_info, bilateral_info, tc_calib = ukraine_war(country_dict = country_info, bilateral_dict = bilateral_info, tc_dict = tc_calib)

    ### Export restrictions ###
    if shock_dict['exportres']=='yes':
        tc_calib = export_restrictions(tc_dict = tc_calib, trade_restriction_df = trade_restrictions_df)

    ### Yield variability ###
    if shock_dict['yieldvar']=='yes':
        country_info = yield_variability(country_dict = country_info, yield_df = yield_var_df, year = year_select)

    return country_info, bilateral_info, trade_calib, prodprice_calib, prodprice_increment, conprice_calib, tc_calib, calib_constant

def estimate_price_increment(prodprice_increment_dict, trade_dict):
    ### add increment to trade df ###
    increment_df = pd.Series(prodprice_increment_dict).reset_index().rename(columns = {'index':'from_iso3',0:'increment'})
    trade_df = pd.DataFrame(pd.Series(trade_dict)).reset_index().rename(columns = {'level_0':'from_iso3','level_1':'to_iso3',0:'trade'})
    trade_df = trade_df.merge(increment_df, on= 'from_iso3')

    ## total increase
    trade_df['increment_amount'] = trade_df['trade']* trade_df['increment']

    ### weighted average increase
    conprice_increment_dict = (trade_df.groupby(['to_iso3'])['increment_amount'].sum()/trade_df.groupby(['to_iso3'])['trade'].sum()).to_dict()

    return conprice_increment_dict

def set_conprice_init(dict_con, dict_increment):
    conprice_init_dict ={}
    for key in dict_con:
        conprice_init_dict[key] = dict_con[key] + dict_increment[key]

    return conprice_init_dict

def shock_trade_clearance(country_info, bilateral_info, eps_val, sigma_val, crop_code, calibration_output_path, shock_dict, year_select, error, error_scale = 100, max_iter = 3000):
    ### read the yield variability data ###
    yield_var_df = pd.read_csv('Input/Production_yield_shock/country_anomaly_fraction_detrended_'+crop_code+'.csv').replace(np.nan, 1).replace(0, 1)
    ### make sure it has a proper lower limit ####
    for column in yield_var_df.set_index('iso3').columns:
        yield_var_df[column] = np.where(yield_var_df[column]<0.25, 0.25, yield_var_df[column])

    ### import and export restrictions data ###
    trade_restrictions = pd.read_csv('Input/Trade_policies/trade_bans_processed.csv')
    trade_restrictions_crop = trade_restrictions[trade_restrictions['crop_code']==crop_code].reset_index(drop = True)

    ### short term elasticity scaling factor ###
    short_term_demand_elas_scaling = 0.7
    short_term_supply_elas_scaling = 1.0
    print(crop_code,year_select,'running', datetime.datetime.now(), max_iter)
    ### original supply and demand ####
    supply_original_dict = (country_info.supply  +  (error/error_scale)*len(country_info.supply)).to_dict()
    demand_original_dict = (country_info.demand  + (error/error_scale)*len(country_info.demand)).to_dict()

    #### read the calibration output #
    trade_calib, prodprice_calib, conprice_calib, tc_calib, calib_constant =  read_calibration_output(calibration_output_path, crop_code)
    ### implement shock  ###
    country_info, bilateral_info, trade_calib, prodprice_calib, prodprice_increment, conprice_calib, tc_calib, calib_constant =implement_shock_scenario(shock_dict = shock_dict,
                                                                                                                                                        country_info = country_info,
                                                                                                                                                        bilateral_info =  bilateral_info,
                                                                                                                                                        trade_calib = trade_calib,
                                                                                                                                                        prodprice_calib = prodprice_calib,
                                                                                                                                                        conprice_calib = conprice_calib,
                                                                                                                                                        tc_calib = tc_calib,
                                                                                                                                                        calib_constant = calib_constant,
                                                                                                                                                        trade_restrictions_df = trade_restrictions_crop,
                                                                                                                                                        yield_var_df = yield_var_df,
                                                                                                                                                        year_select = year_select)


    ### estimate price increment
    conprice_increment_dict = estimate_price_increment(prodprice_increment_dict = prodprice_increment, trade_dict = trade_calib)
    ## new consumer price ###
    conprice_init =  set_conprice_init(dict_con = conprice_calib, dict_increment = conprice_increment_dict)

    ####-------- INITIALIZE THE MODEL --------######
    model2 =  ConcreteModel()
    model2.i = Set(initialize=country_info.iso3, doc='Countries')

    ####-------- PARAMETERS --------######
    model2.prodprice03 = Param(model2.i, initialize= prodprice_calib, doc='production price 03')

    ### increment due to price increase ###
    model2.price_increment = Param(model2.i, initialize= prodprice_increment, doc='production price increment')
    model2.conprice03 = Param(model2.i,initialize= conprice_calib, doc='consumer price 03')
    model2.tc03 = Param(model2.i, model2.i, initialize= tc_calib, doc='transportation cost 03')
    model2.calib = Param(model2.i, model2.i, initialize= calib_constant, doc='transportation cost 03')

    ### tariffs
    model2.adv = Param(model2.i, model2.i, initialize=bilateral_info.adv.to_dict(), doc='tariff')
    #model2.alpha = Param(initialize = 0.5)

    #### Demand and Supply Elasticities ####
    model2.Ed = Param(model2.i,initialize=(country_info.demand_elas* short_term_demand_elas_scaling).to_dict(), doc='demand elasticity') ### divided by 1,000
    model2.Es = Param(model2.i,initialize=(country_info.supply_elas * short_term_supply_elas_scaling).to_dict(), doc='supply elasticity') ### divided by 1,000

    ### baseline demand and supply ###
    model2.demand_original = Param(model2.i,initialize = demand_original_dict, doc='demand initial')
    model2.supply_original = Param(model2.i, initialize = supply_original_dict, doc='supply initial')

    model2.demand03 = Param(model2.i,initialize=(country_info.demand  +(error/error_scale)*len(country_info.demand)).to_dict(), doc='demand initial')
    model2.supply03 = Param(model2.i, initialize=(country_info.supply +(error/error_scale)*len(country_info.supply)).to_dict(), doc='supply initial')
    model2.stocks = Param(model2.i, initialize=(country_info.stocks), doc = 'stocks')
    model2.stocks_min = Param(model2.i, initialize=(country_info.stocks_min), doc = 'stocks')

    ### set parameters ##
    model2.epsilon = Param(initialize=0.001, doc='eps')
    model2.eps = Param(initialize=eps_val, doc='eps')
    model2.sigma = Param(initialize=sigma_val, doc='sigma')
    model2.existing_trade_binary = Param(model2.i, model2.i, initialize=bilateral_info.trade_binary.to_dict(), doc='binary existing trade')
    model2.existing_trade = Param(model2.i, model2.i, initialize = bilateral_info.trade_old.to_dict(), doc='existing trade')

    #### shock and total supply ####
    def supply_shock(model2, i):
        return model2.supply_original[i] - model2.supply03[i]

    model2.supply_shock = Param(model2.i, initialize=supply_shock)

    def eq_supply_total(model2, i):
        return model2.supply03[i] + (model2.stocks[i]-model2.stocks_min[i])

    model2.supply_total = Param(model2.i, initialize =eq_supply_total)

    #### Create the demand and supply curves #####
    def B(model2, i):
        if model2.demand_original[i]>1:
            return (model2.conprice03[i])/(model2.demand_original[i]*model2.Ed[i])
        else:
            ### take average
            B_sum = 0; count = 0
            for j in model2.i:
                if model2.demand_original[j]>1:
                    B_sum+=(model2.conprice03[j])/(model2.demand_original[j]*model2.Ed[j])
                    count+=1

            return B_sum/count

    def D(model2, i):
        if model2.supply_original[i]>1:
            return model2.prodprice03[i]/(model2.supply_original[i]*model2.Es[i])
        else:
            ### take average
            D_sum = 0; count = 0
            for j in model2.i:
                if model2.supply_original[j]>1:
                    D_sum+=model2.prodprice03[j]/(model2.supply_original[j]*model2.Es[j])
                    count+=1

            return D_sum/count

    def A(model2, i):
        if model2.demand_original[i]>1:
            return (model2.conprice03[i]) + model2.B[i] * model2.demand_original[i]
        else:
            return (model2.conprice03[i]) + model2.B[i] * model2.demand_original[i] - model2.epsilon

    def C(model2, i):
        if model2.supply_original[i]>1:
            return model2.prodprice03[i] - model2.D[i] * model2.supply_original[i]
        else:
            return model2.prodprice03[i] - model2.D[i] * model2.supply_original[i] + model2.epsilon

    model2.B = Param(model2.i,initialize=B, doc='B: Absolute value of the inverse demand function slopes p=f(q)')
    model2.A = Param(model2.i,initialize=A, doc='A: Inverse demand function intercepts p=f(q)')

    model2.D = Param(model2.i,initialize=D, doc='D: Absolute value of the inverse supply function slopes p=f(q)')
    model2.C = Param(model2.i,initialize=C, doc='C: Inverse supply function intercepts p=f(q)')

    def tariff_initialize(model2, i, j):
        return (model2.prodprice03[i] + model2.price_increment[i] + model2.tc03[i,j]-model2.calib[i,j]) * model2.adv[i,j]

    model2.tariff3 = Param(model2.i, model2.i, initialize = tariff_initialize, doc='tariff')

    ####-------- VARIABLES --------######
    def prodprice_bounds(model2, i):
        return (0, 1.01*(model2.C[i] + (model2.supply_shock[i]*model2.D[i]) + model2.D[i] * model2.supply_total[i]))

    model2.prodprice3 = Var(model2.i,initialize = model2.prodprice03.extract_values(), bounds = prodprice_bounds, doc='production price 3')
    model2.conprice3 = Var(model2.i,initialize = model2.conprice03.extract_values(),  within = PositiveReals, doc='consumer price 3')
    model2.trade3 = Var(model2.i, model2.i, initialize= trade_calib,  bounds = (error/error_scale, None), doc='trade3')

    model2.demand = Var(model2.i,initialize=model2.demand03.extract_values(),  bounds = (len(country_info.demand)*(error/error_scale), None), doc='demand')

    def supply_bounds(model2, i):
        return (len(country_info.demand)*(error/error_scale), model2.supply_total[i])

    model2.supply = Var(model2.i,initialize=model2.supply03.extract_values(),  bounds = supply_bounds, doc='supply')

    ####-------- EQUATIONS --------######
    def eq_PROD(model2, i):
            return complements(model2.supply[i] == sum(model2.trade3[i,j] for j in model2.i), model2.prodprice3[i]>0)

    def eq_DEM(model2, i):
            return complements(sum(model2.trade3[j,i] for j in model2.i) == model2.demand[i], model2.conprice3[i]>0)

    def eq_DPRICEDIF(model2, i):
            return complements(model2.conprice3[i] >= model2.A[i] - model2.B[i] * model2.demand[i], model2.demand[i]>=len(country_info.demand)*(error/error_scale))
            #return complements(model2.demand[i] == model2.demand_original[i] * pow(model2.conprice3[i]/model2.conprice03[i], -model2.Ed[i]), model2.demand[i]>=len(country_info.demand)*(error/error_scale))

    def eq_SPRICEDIF(model2, i):
            return complements(model2.C[i] + (model2.supply_shock[i]*model2.D[i]) + model2.D[i] * model2.supply[i] >= model2.prodprice3[i] , model2.supply[i]<=model2.supply_total[i])
            #return complements( model2.supply[i] == model2.supply_original[i] * pow(model2.prodprice3[i]/model2.prodprice03[i], model2.Es[i]) , model2.supply[i]<=model2.supply03[i])

    def eq_PRLINK2(model2, i, j):
        if model2.existing_trade_binary[i,j]==1:
            return complements(model2.prodprice3[i] + model2.price_increment[i] + model2.calib[i,j]+ model2.tariff3[i,j] + (model2.tc03[i,j]-model2.calib[i,j]) * pow(model2.trade3[i,j]/model2.existing_trade[i,j], 1/model2.eps) >= model2.conprice3[j], model2.trade3[i,j]>=(error/error_scale))
        else:
            return complements(1.2* model2.prodprice3[i] + model2.price_increment[i]+ model2.calib[i,j] + model2.tariff3[i,j]+ (model2.tc03[i,j]-model2.calib[i,j]) + model2.sigma * model2.trade3[i,j]  >= model2.conprice3[j], model2.trade3[i,j]>=(error/error_scale))


    ### add constraints
    model2.eq_PROD = Complementarity(model2.i, rule = eq_PROD, doc='Supply >= quantity shipped')
    model2.eq_DEM = Complementarity(model2.i, rule = eq_DEM, doc='Demand <= quantity shipped')
    model2.eq_DPRICEDIF = Complementarity(model2.i, rule = eq_DPRICEDIF, doc='difference market demand price and local demand price')
    model2.eq_SPRICEDIF = Complementarity(model2.i, rule = eq_SPRICEDIF, doc='difference market supply price and local supply price')
    model2.eq_PRLINK2 = Complementarity(model2.i, model2.i, rule = eq_PRLINK2, doc='price chain 2')

    ####-------- SOLVE --------######
    TransformationFactory('mpec.simple_nonlinear').apply_to(model2)

    ### choose solver
    opt = SolverFactory('ipopt', solver_io='nl')
    opt.options['linear_solver'] = 'ma27'
    opt.options['nlp_scaling_method'] = 'user-scaling'
    opt.options['tol'] = 1
    opt.options['acceptable_tol'] = 1
    opt.options['max_iter'] = max_iter
    opt.options['max_cpu_time'] = 60 * 20 ### 20 min##
    opt.options['hsllib'] = 'libcoinhsl.dylib'


    result=opt.solve(model2)

    initial_supply = np.sum(list(model2.supply03.extract_values().values()))/1e6
    initial_demand = np.sum(list(model2.demand03.extract_values().values()))/1e6
    output_supply =  np.sum(list(model2.supply.extract_values().values()))/1e6
    output_demand =  np.sum(list(model2.demand.extract_values().values()))/1e6
    print(year_select, initial_supply,initial_demand, output_supply, output_demand)
    ### process output
    trade, country_output = process_final_output(model = model2, year_select= year_select, crop_code = crop_code, error = error, error_scale = error_scale)



    return trade, country_output
#
