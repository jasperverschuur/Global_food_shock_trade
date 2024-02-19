# Global food shock and trade model
Main code to run global food shock and agricultural trade model, supporting the results of Verschuur et al. (under review), "The impacts of polycrises on global grain availability and prices". 

This includes:
1/ The calibration procedure per crop to quantify the calibration constants per trade flow given the present-day trade network
2/ The simulation analysis of various shocks, including:
-(i) 54 years of baseline yield variability;
-(ii) (i) + Ukraine war supply shock;
-(iii) (i) + Energy price shock;
-(iv) (i) + Trade bans
-(v) (i) + all shocks combined

All optimisation is written in Pyomo with HSL's MA27 as (non-linear) solver. 

