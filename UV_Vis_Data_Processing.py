## This code is for the determination of phenolic concentrations and colour measurements from UV-Visible spectroscopy raw data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import os 

RawData = pd.read_excel('RawDataCondensed.xlsx', sheet_name='Raw Data samples') # Import data from excel sheet
CD = pd.read_excel('RawDataCondensed.xlsx', sheet_name='ColourDensity')         # Colour density data
HCl = pd.read_excel('RawDataCondensed.xlsx', sheet_name='1MHCl')                # Anthocyanin content data
BSL = pd.read_excel('RawDataCondensed.xlsx', sheet_name='BSL')                  # SO2 resistant pigments content data
MCP = pd.read_excel('RawDataCondensed.xlsx', sheet_name='MCP')                  # Tannin content data

#colour density calculation
lambda_420 = CD.loc[110]                                  # Absorbances at 420 nm
lambda_520 = CD.loc[160]                                  # Absorbances at 520 nm
lambda_620 = CD.loc[210]                                  # Absorbances at 620 nm
colour_density_all = lambda_420 + lambda_520 + lambda_620 # The colour density is determined as the sum of absorbances
                                                          # at 420 nm, 520 nm and 620 nm
colour_density = colour_density_all[1:]
#print(type(colour_density))

# Total anthocyanin and total phenolic content
Iland_280_all = HCl.loc[40]                           # Abosrbances at 520 nm
Iland_280 = Iland_280_all[1:] 
#print(type(Iland_280))                               #absorbtion values at 280nm

DF = 51 #dilution factor
TPI = Iland_280 * DF                                  #total phenolics index is cacluated as the absorbances at 280 nm multiplied by DF
Iland_520_all = HCl.loc[160]                          # absorbance values at 520nm
Iland_520 = Iland_520_all[1:]  

MW = 529                                              #molecular weight in g/mol of malvin-3-glucoside
E = 28000                                             #extinction coefficient in L/cm.mol of malvin-3-glucoside
L = 1                                                 #standard 1cm pathlength

anthocyanins = ((Iland_520 * MW * DF) / (E * L))*1000 # calculates total anthocyanin content in mg/L

#SO2 resistant pigments calculation
Somers_520_all = BSL.loc[160]                                    # The absorbance at 520 nm for SO2 resistant pigments
Somers_520 = Somers_520_all[1:]
SO2_resistant_pigments = ((Somers_520 * MW * DF) / (E * L))*1000 # Calculates total SO2 resistant pigment content in mg/L

colour_density_df = pd.Series(colour_density, name='Colour Density') # Converting dataframe to panda series and renaming 
anthocyanin_df = pd.Series(anthocyanins, name='Anthocyanin Content') # Converting dataframe to panda series and renaming
SO2_resistant_pigments_df = pd.Series(SO2_resistant_pigments, name='SO2 Resistant Pigments') # Converting dataframe to panda series
                                                                                             # and renaming
UV_vis_df = pd.concat([colour_density_df, anthocyanin_df, SO2_resistant_pigments_df],axis=1) # Combining all columns into a dataframe

RawData.index += 1 
RawData_merge = RawData.join(UV_vis_df) # Combining UV-Visble concentration data to raw data

#MCP data
MCP_diff1 = MCP["C1"]-MCP["T1"] # Treatment sample absorbances subtracted from control sample absorbances for repeat 1
MCP_diff2 = MCP["C2"]-MCP["T2"] # Treatment sample absorbances subtracted from control sample absorbances for repeat 2
MCP_diff3 = MCP["C3"]-MCP["T3"] # Treatment sample absorbances subtracted from control sample absorbances for repeat 3
MCP_diff1_df = pd.Series(MCP_diff1, name='C1-T1') # Coonverting repeat 1 data to panda series
MCP_diff2_df = pd.Series(MCP_diff2, name='C2-T2') # Coonverting repeat 2 data to panda series
MCP_diff3_df = pd.Series(MCP_diff3, name='C3-T3') # Coonverting repeat 3 data to panda series
MCP_df = pd.concat([MCP_diff1_df, MCP_diff2_df, MCP_diff3_df],axis=1) # Combining MCP data into a dataframe

MCP_df['Average'] = MCP_df.mean(axis=1)                     # Calculates average of three repeats
cols = ['C1-T1', 'C2-T2', 'C3-T3']                          # Naming columns of repeats
MCP_df['Stdev'] = MCP_df[cols].std(axis=1)                  # Calculates standard deviation of repeats
MCP_df['Error'] = (MCP_df['Stdev'] / MCP_df['Average'])*100 # Calculates error of repeats
#print(MCP_df)

error_indexes = np.where(MCP_df['Error'] > 11)[0] # Determines the indexes where the error is greater than 11
#type(error_indexes)
#print(MCP_df.iloc[140])

def calc_error(val1, val2):
    # This function calculates the average, error and standard deviation of only 2 of the three repeats
    # and returns the average and error
    Average = np.mean([val1, val2])
    Stdev = np.std([val1, val2])
    Error = (Stdev / Average)*100
    
    return [Average, Error]

def error_resolve(row):
    #This functions calculates which column should be left out to achieve
    # an error <= 11 (if possible) 
    ar = []
    cols = ['C1-T1', 'C2-T2', 'C3-T3']
    for i in range(0, len(cols)):
        for j in range(i + 1, len(cols)):
            Average, Error = calc_error(row[cols[i]], row[cols[j]])
            
            #print(f'val1 [{cols[i]}]: {row[cols[i]]}')
            #print(f'val2 [{cols[j]}]: {row[cols[j]]}')
            #print(f"Average: {Average}")
            #print(f"Error: {Error}")
            
            ar.append([Average, Error])  
        
    min_index = np.argmin([ar[0][1], ar[1][1], ar[2][1]])
    
    return ar[min_index]

for error_row in error_indexes:
    row = MCP_df.iloc[error_row]
    #print("row")
    #print(row)
    Average, Error = error_resolve(row)
    #print(Average)
    #print(Error)
    MCP_df.at[error_row, 'Average'] = Average # appends new average values to dataframe to replace indexes where error is greater than 11
    MCP_df.at[error_row, 'Error'] = Error     # appends new error values to dataframe to replace indexes where error is greater than 11
    
    #print("\n\n")

#MCP Tannin calibration curve
a = 6.4497847025 # gradient of tannin calibration curve 
b = 0.0224074127 # intercept of tannin calibration curve
DF_MCP = 40      # Dilution factor for MCP tannins

Tannins_conc = (a * MCP_df['Average'] - b) * DF_MCP # calculation of tannin concentrations
#print(Tannins_conc)
Tannins_conc.index += 1

RawData_merge['Tannins'] = Tannins_conc # Append tannin concentration data to dataframe