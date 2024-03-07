# Global food shock and trade model
Main code to run global food shock and agricultural trade model, supporting the results of Verschuur et al. (under review), "The impacts of polycrises on global grain availability and prices". <br>

This includes: <br>
1/ The calibration procedure per crop to quantify the calibration constants per trade flow given the present-day trade network <br>
2/ The simulation analysis of various shocks, including: <br>
-(i) 54 years of baseline yield variability; <br>
-(ii) (i) + Ukraine war supply shock;<br>
-(iii) (i) + Energy price shock;<br>
-(iv) (i) + Trade bans;<br>
-(v) (i) + all shocks combined<br>

<br>

All optimisation is written in Pyomo (https://www.pyomo.org/) with HSL's MA27 as (non-linear) solver: <br>
https://github.com/coin-or-tools/ThirdParty-HSL<br>
Alternatively, one can run the code using the standard ipopt solver in Pyomo. <br>
<br>
Software requirements:<br>
-Python Python 3.9.12<br>
-Tested on MacOS Monterey v12.2.1<br>
<br>
Runtime:<br>
-Calibration runtime depends on crop but around 1-2h.<br>
-To run the shocks, runtime is around 15 minutes per modelled year. <br>
<br>
Demo:<br>
Alongside the code for the paper, we uploaded a small demo model based on the AGRODEP Spatial trade model developed by IFPRI (https://www.agrodep.org/models/agrodep-spatial-equilibrium-model), which can help users to get familiar with spatial price equilibrium modelling. We have modified the original AGRODEP model to be in line with the trade cost formulation adopted in our paper. 
Runtime should be minutes. 


