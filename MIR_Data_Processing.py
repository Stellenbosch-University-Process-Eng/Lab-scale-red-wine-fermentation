import UV_Vis_Data_Processing

#Importing MIR data

import spectrochempy as scp
from brukeropusreader import read_file
import pandas as pd
import numpy as np
from matplotlib import pylab as plt
import os

#get wavelength from one opus data file

opus10 = read_file("TJ_24__EVAL.0")
abs10 = opus10["AB"]
FXV = opus10["AB Data Parameter"]["FXV"] # minimum wavelength
LXV = opus10["AB Data Parameter"]["LXV"] # maximum wavelength
wavelength = np.linspace(FXV,LXV,880)

#Get MIR data from Opus files
directory = 'C:/Users/thaak/Stellenbosch University/All Data'
IR_data = {}
for opusfile in os.listdir(directory):
    
    if opusfile.endswith(".0"):
        opusfile_data = read_file(opusfile)
        file_absorption_data = np.array(opusfile_data["AB"])
        IR_data[opusfile] = file_absorption_data
    else:
        pass

#for key in IR_data:
    #print(key)
    #print(IR_data[key])
    #plt.plot(wavelength, IR_data[key])
#plt.show()

IR_data_Df = pd.DataFrame.from_dict(IR_data) # Converting MIR data to dataframe
X = IR_data_Df.T # Transposing MIR data

#Sort MIR data in order from sample 1 to 168 (TJ_1__EVAL.0 - TJ_168__EVAL.0)
import re
a_sort = sorted(X.index, key=lambda s: int(re.search(r'\d+', s).group()))
#print(a_sort)
X = X.reindex(index=a_sort)

#Applying PCA to MIR data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from UV_Vis_Data_Processing import RawData_merge

scalar = StandardScaler() # Calculating the standardizing parameters and transforming the dataset.  
X_scaled = scalar.fit_transform(X) 
pca = PCA() 
T = pca.fit_transform(X_scaled) # Applying PCA

pca_var = pd.DataFrame(pca.explained_variance_ratio_) # Variances explained by principal components

from UV_Vis_Data_Processing import RawData_merge
from UV_Vis_Data_Processing import TPI
RawData_merge["TPI"] = TPI # Add TPI to dataframe
import numpy as np
import pandas as pd

RawData_merge['Anthocyanin Content'][12:19] = np.nan
RawData_merge['Anthocyanin Content'][42] = np.nan
RawData_merge['Anthocyanin Content'][68] = np.nan
RawData_merge['TPI'][12:19] = np.nan
RawData_merge['TPI'][42] = np.nan
RawData_merge['TPI'][68] = np.nan

RawData_merge = RawData_merge.interpolate()

#Linear regression and cross validation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn import metrics

def regress_cross_val(m, y):
    #This function applies linear regression and cross validation using m principal   
    #components
    #and estimates/returns the cross validation test error
    
    Z = T[:,:m] #Let predictors Z be the first m principal components 
               #(that is, Z is the first m columns of T)
    
    lm = LinearRegression()
    model = lm.fit(Z, y) # Applying linear regression
    cv = KFold(n_splits=10, random_state=1, shuffle=True) # Applying k-fold cross 
                                                          # validation
    y_pred = cross_val_predict(model, Z, y, cv=cv)  # Determining the predicted 
                                                    # concentrations from the model
    #print(f"y_pred: {y_pred}")
    
    return y_pred, Z
         
def min_test_error(y):
    #This function selects m(no. of principal components) which results in the lowest
    #cross-validation test error
    
    min_index = -1
    min_err = 1000000000
    for m1 in range(1, 168):
        y_pred, Z = regress_cross_val(m1, y)
        test_error = np.mean(np.abs((y - y_pred)/y))*100 # Calulates minimum MAPE
        
        if test_error < min_err:
            min_index = m1
            min_err = test_error
            
    max_index = -1
    max_err = 0
    for m2 in range(1, 168):
        y_pred, Z = regress_cross_val(m2, y)
        test_error = np.mean(np.abs((y - y_pred)/y))*100 # Calulates maximum MAPE
        
        if test_error > max_err:
            max_index = m2
            max_err = test_error
    
    return min_err, max_err, min_index, max_index

# Display minimum mand maximum MAPE values as well as the number of principal components
concentrations = ["Colour Density", "Anthocyanin Content", "Tannins", "SO2 Resistant Pigments", "TPI"]

for concentration in concentrations:
    #print(concentration+ ": ")
    min_err, max_err, min_index, max_index = min_test_error(RawData_merge[concentration])
    #print(f"Minimum test error: {min_err}")
    #print(f"No. of principal components: {min_index}")
    #print(f"Maximum test error: {max_err}")
    #print(f"No. of principal components: {max_index}")
    
    #print("")

# Combining the minimum MAPE values and corresponding number of principal components into 1 
# dataframe for each model

# Colour density
mte_CD = []
for m in range(1, 149):
    y_pred, Z = regress_cross_val(m, RawData_merge["Colour Density"])
    test_error = mean_absolute_error(RawData_merge["Colour Density"], y_pred)
    mte_CD.append([m, test_error])
mte_CD_df = pd.DataFrame(mte_CD, columns = ['m','test_error'])    


mte_Anth = []
for m in range(1, 149):
    y_pred, Z = regress_cross_val(m, RawData_merge["Anthocyanin Content"])
    test_error = mean_absolute_error(RawData_merge["Anthocyanin Content"], y_pred)
    mte_Anth.append([m, test_error])
mte_Anth_df = pd.DataFrame(mte_Anth, columns = ['m','test_error'])    

# TPI
mte_TPI = []
for m in range(1, 149):
    y_pred, Z = regress_cross_val(m, RawData_merge["TPI"])
    test_error = mean_absolute_error(RawData_merge["TPI"], y_pred)
    mte_TPI.append([m, test_error])
mte_TPI_df = pd.DataFrame(mte_TPI, columns = ['m','test_error'])    

# Tannin content
mte_Tann = []
for m in range(1, 149):
    y_pred, Z = regress_cross_val(m, RawData_merge["Tannins"])
    test_error = mean_absolute_error(RawData_merge["Tannins"], y_pred)
    mte_Tann.append([m, test_error])
mte_Tann_df = pd.DataFrame(mte_Tann, columns = ['m','test_error'])    

# SO2 resistant polymeric pigment content
mte_SO2 = []
for m in range(1, 149):
    y_pred, Z = regress_cross_val(m, RawData_merge["SO2 Resistant Pigments"])
    test_error = mean_absolute_error(RawData_merge["SO2 Resistant Pigments"], y_pred)
    mte_SO2.append([m, test_error])
mte_SO2_df = pd.DataFrame(mte_SO2, columns = ['m','test_error'])    

#convert dataframe from a wide to long form 
mte_all_dfm = mte_all_df.melt('m', var_name='Model', value_name='Test Errors')
print(mte_all_dfm)

# Plotting change in MAPE versus number of principal components
import seaborn as sns
sns.set_style("white")

sns.lineplot(x='m', y='Test Errors', hue='Model', data=mte_all_dfm, lw=2, palette='deep')
plt.ylabel('Cross-Validated MAPE (%)', fontsize = 15)
plt.xlabel('No. of Principal Components', fontsize = 15)
plt.legend(fontsize = 12)
plt.xticks(fontsize = 12)
plt.xlim(0,150)
plt.ylim(0,200)
plt.yticks(fontsize = 12)
plt.show()

# Assigning variabales to actual and predicted measurements
y1 = RawData_merge["Colour Density"] 
y_pred1, Z1 = regress_cross_val(9, y1)

y2 = RawData_merge["Anthocyanin Content"]
y_pred2, Z2 = regress_cross_val(9, y2)

y3 = RawData_merge["Tannins"]
y_pred3, Z3 = regress_cross_val(12, y3)

y4 = RawData_merge["SO2 Resistant Pigments"]
y_pred4, Z4 = regress_cross_val(8, y4)

y5 = RawData_merge["TPI"]
y_pred5, Z5 = regress_cross_val(13, y5)

# Create one dataframe of all actual and predicted values for colour density, and phenolics

conc_MIR = [y_pred1, y_pred2, y_pred5, y_pred3, y_pred4]                     # All predicted MIR values
MIR_conc = (pd.DataFrame(conc_MIR)).T                                        # Transpose 
MIR_conc.columns = ['MIR_CD', 'MIR_Antho', 'MIR_TPI', 'MIR_Tann', 'MIR_SO2'] # Name columns
MIR_conc.index += 1

conc_UVVis = [y1, y2, y5, y3, y4]                                                        # All UV-Vis determined concentrations (actual)
UVVis_conc = (pd.DataFrame(conc_UVVis)).T                                                # Transpose
UVVis_conc.columns = ['UVVis_CD', 'UVVis_Antho', 'UVVis_TPI', 'UVVis_Tann', 'UVVis_SO2'] # Name columns

UVVis_MIR_conc = UVVis_conc.join(MIR_conc) # Combine predicted and actual values into one dataframe

ferment = RawData_merge["Descr"] # Add a description column 

UVVis_MIR_conc["Description"] = ferment
UVVis_MIR_conc["Day"] = RawData_merge["Day"] # Add column for day of fermentation