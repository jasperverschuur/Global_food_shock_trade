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

All optimisation is written in Pyomo (https://www.pyomo.org/) with HSL's MA27 as (non-linear) solver:
https://github.com/coin-or-tools/ThirdParty-HSL
Alternatively, one can run the code using the standard ipopt solver in Pyomo. 

Software requirements:
-Python Python 3.9.12
-Tested on MacOS Monterey v12.2.1

Runtime:
-Calibration runtime depends on crop but around 1-2h.
-To run the shocks, runtime is around 15 minutes per modelled year. 

Demo:
Alongside the code for the paper, we uploaded a small demo model based on the AGRODEP Spatial trade model developed by IFPRI (https://www.agrodep.org/models/agrodep-spatial-equilibrium-model), which can help users to get familiar with spatial price equilibrium modelling. We have modified the original AGRODEP model to be in line with the trade cost formulation adopted in our paper. 
Runtime should be minutes. 


