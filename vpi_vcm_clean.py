import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from pathlib import Path

root = Path('~/PycharmProjects/vpi_vcm_processing/').expanduser()

df = pd.read_csv("resources/Vpi_Dec_22_sep2.csv", sep=';')

df_lambda1 = df.loc[df["WAVELENGTH_NM"] == 1528.5]
df_lambda2 = df.loc[df["WAVELENGTH_NM"] == 1564]

# build empty dataframe
df_vcm_lambda = pd.DataFrame(columns=["VCM_LAMBDA", "Vpi_MZ1", "Vpi_MZ2", "Vpi_MZ3", "Vpi_MZ4"])
x_new = np.linspace(1528.5, 1564.0, 15641-15285).round(1)
lambdas = [1528.5, 1564]
vcm_list = ["VCM1", "VCM2", "VCM3", "VCM4"]
mz_list = ["MZ1", "MZ2", "MZ3", "MZ4"]
labels_vcm1 = ["VCM1_" + str(x_new[j]) for j in np.arange(np.size(x_new))]
labels_vcm2 = ["VCM2_" + str(x_new[j]) for j in np.arange(np.size(x_new))]
labels_vcm3 = ["VCM3_" + str(x_new[j]) for j in np.arange(np.size(x_new))]
labels_vcm4 = ["VCM4_" + str(x_new[j]) for j in np.arange(np.size(x_new))]

vcm_lambda_array = [str(j + "_" + str(i)) for i in x_new for j in vcm_list]
df_vcm_lambda["VCM_LAMBDA"] = vcm_lambda_array

vcm1_indexes = [df_vcm_lambda[df_vcm_lambda["VCM_LAMBDA"] == label].index.values.astype(int)[0] for label in labels_vcm1]
vcm2_indexes = [df_vcm_lambda[df_vcm_lambda["VCM_LAMBDA"] == label].index.values.astype(int)[0] for label in labels_vcm2]
vcm3_indexes = [df_vcm_lambda[df_vcm_lambda["VCM_LAMBDA"] == label].index.values.astype(int)[0] for label in labels_vcm3]
vcm4_indexes = [df_vcm_lambda[df_vcm_lambda["VCM_LAMBDA"] == label].index.values.astype(int)[0] for label in labels_vcm4]

# VCM1
vpi_double_mz1_lambda1_vcm1 = df_lambda1["MZ1_RF_DIFF_VCM1_2VPI_V"]
vpi_double_mz2_lambda1_vcm1 = df_lambda1["MZ2_RF_DIFF_VCM1_2VPI_V"]
vpi_double_mz3_lambda1_vcm1 = df_lambda1["MZ3_RF_DIFF_VCM1_2VPI_V"]
vpi_double_mz4_lambda1_vcm1 = df_lambda1["MZ4_RF_DIFF_VCM1_2VPI_V"]

vpi2_mz1_lambda1_vcm1_arr = [vpi_double_mz1_lambda1_vcm1.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz1_lambda1_vcm1))
                             if vpi_double_mz1_lambda1_vcm1.iloc[i] > 2]
vpi2_mz2_lambda1_vcm1_arr = [vpi_double_mz2_lambda1_vcm1.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz2_lambda1_vcm1))
                             if vpi_double_mz2_lambda1_vcm1.iloc[i] > 2]
vpi2_mz3_lambda1_vcm1_arr = [vpi_double_mz3_lambda1_vcm1.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz3_lambda1_vcm1))
                             if vpi_double_mz3_lambda1_vcm1.iloc[i] > 2]
vpi2_mz4_lambda1_vcm1_arr = [vpi_double_mz4_lambda1_vcm1.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz4_lambda1_vcm1))
                             if vpi_double_mz4_lambda1_vcm1.iloc[i] > 2]

vpi2_mz1_lambda1_vcm1 = np.mean(vpi2_mz1_lambda1_vcm1_arr)
vpi2_mz2_lambda1_vcm1 = np.mean(vpi2_mz2_lambda1_vcm1_arr)
vpi2_mz3_lambda1_vcm1 = np.mean(vpi2_mz3_lambda1_vcm1_arr)
vpi2_mz4_lambda1_vcm1 = np.mean(vpi2_mz4_lambda1_vcm1_arr)

vpi_mz1_lambda1_vcm1 = vpi2_mz1_lambda1_vcm1/2
vpi_mz2_lambda1_vcm1 = vpi2_mz2_lambda1_vcm1/2
vpi_mz3_lambda1_vcm1 = vpi2_mz3_lambda1_vcm1/2
vpi_mz4_lambda1_vcm1 = vpi2_mz4_lambda1_vcm1/2

vpi2_lambda1_vcm1_mean = np.mean([vpi2_mz1_lambda1_vcm1, vpi2_mz2_lambda1_vcm1,
                                  vpi2_mz3_lambda1_vcm1, vpi2_mz4_lambda1_vcm1])

vpi_lambda1_vcm1_mean = vpi2_lambda1_vcm1_mean/2

vpi_double_mz1_lambda2_vcm1 = df_lambda2["MZ1_RF_DIFF_VCM1_2VPI_V"]
vpi_double_mz2_lambda2_vcm1 = df_lambda2["MZ2_RF_DIFF_VCM1_2VPI_V"]
vpi_double_mz3_lambda2_vcm1 = df_lambda2["MZ3_RF_DIFF_VCM1_2VPI_V"]
vpi_double_mz4_lambda2_vcm1 = df_lambda2["MZ4_RF_DIFF_VCM1_2VPI_V"]

vpi2_mz1_lambda2_vcm1_arr = [vpi_double_mz1_lambda2_vcm1.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz1_lambda2_vcm1))
                             if vpi_double_mz1_lambda2_vcm1.iloc[i] > 2]
vpi2_mz2_lambda2_vcm1_arr = [vpi_double_mz2_lambda2_vcm1.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz2_lambda2_vcm1))
                             if vpi_double_mz2_lambda2_vcm1.iloc[i] > 2]
vpi2_mz3_lambda2_vcm1_arr = [vpi_double_mz3_lambda2_vcm1.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz3_lambda2_vcm1))
                             if vpi_double_mz3_lambda2_vcm1.iloc[i] > 2]
vpi2_mz4_lambda2_vcm1_arr = [vpi_double_mz4_lambda2_vcm1.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz4_lambda2_vcm1))
                             if vpi_double_mz4_lambda2_vcm1.iloc[i] > 2]

vpi2_mz1_lambda2_vcm1 = np.mean(vpi2_mz1_lambda2_vcm1_arr)
vpi2_mz2_lambda2_vcm1 = np.mean(vpi2_mz2_lambda2_vcm1_arr)
vpi2_mz3_lambda2_vcm1 = np.mean(vpi2_mz3_lambda2_vcm1_arr)
vpi2_mz4_lambda2_vcm1 = np.mean(vpi2_mz4_lambda2_vcm1_arr)

vpi_mz1_lambda2_vcm1 = vpi2_mz1_lambda2_vcm1/2
vpi_mz2_lambda2_vcm1 = vpi2_mz2_lambda2_vcm1/2
vpi_mz3_lambda2_vcm1 = vpi2_mz3_lambda2_vcm1/2
vpi_mz4_lambda2_vcm1 = vpi2_mz4_lambda2_vcm1/2

vpi2_lambda2_vcm1_mean = np.mean([vpi2_mz1_lambda2_vcm1, vpi2_mz2_lambda2_vcm1,
                                  vpi2_mz3_lambda2_vcm1, vpi2_mz4_lambda2_vcm1])

vpi_lambda2_vcm1_mean = vpi2_lambda2_vcm1_mean/2

x_new1 = np.linspace(1528.5, 1564.0, 15641-15285)
x1 = [1528.5, 1564]
y1 = [vpi2_lambda1_vcm1_mean, vpi2_lambda2_vcm1_mean]
f1 = interpolate.interp1d(x1,y1)
y_new1 = f1(x_new1)

vpi2_lambda_center_vcm1_mean = y_new1[np.where(x_new1 == 1550.0)]
vpi_lambda_center_vcm1_mean = vpi2_lambda_center_vcm1_mean/2

plt.plot(x_new1, y_new1)
plt.show(block=False)

# build VCM1 Vpi data structure to fill the vcm_lambda_df
vpi_mzi_lambda1_vcm1 = [vpi_mz1_lambda1_vcm1, vpi_mz2_lambda1_vcm1, vpi_mz3_lambda1_vcm1, vpi_mz4_lambda1_vcm1]
vpi_mzi_lambda2_vcm1 = [vpi_mz1_lambda2_vcm1, vpi_mz2_lambda2_vcm1, vpi_mz3_lambda2_vcm1, vpi_mz4_lambda2_vcm1]

for i in np.arange(np.size(mz_list)):
    mz_label = 'Vpi_'+mz_list[i]
    y = [vpi_mzi_lambda1_vcm1[i], vpi_mzi_lambda2_vcm1[i]]
    f = interpolate.interp1d(lambdas, y)
    y_new = f(x_new)
    df_vcm_lambda[mz_label].iloc[vcm1_indexes] = y_new

# VCM2
vpi_double_mz1_lambda1_vcm2 = df_lambda1["MZ1_RF_DIFF_VCM2_2VPI_V"]
vpi_double_mz2_lambda1_vcm2 = df_lambda1["MZ2_RF_DIFF_VCM2_2VPI_V"]
vpi_double_mz3_lambda1_vcm2 = df_lambda1["MZ3_RF_DIFF_VCM2_2VPI_V"]
vpi_double_mz4_lambda1_vcm2 = df_lambda1["MZ4_RF_DIFF_VCM2_2VPI_V"]

vpi2_mz1_lambda1_vcm2_arr = [vpi_double_mz1_lambda1_vcm2.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz1_lambda1_vcm2))
                             if vpi_double_mz1_lambda1_vcm2.iloc[i] > 2]
vpi2_mz2_lambda1_vcm2_arr = [vpi_double_mz2_lambda1_vcm2.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz2_lambda1_vcm2))
                             if vpi_double_mz2_lambda1_vcm2.iloc[i] > 2]
vpi2_mz3_lambda1_vcm2_arr = [vpi_double_mz3_lambda1_vcm2.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz3_lambda1_vcm2))
                             if vpi_double_mz3_lambda1_vcm2.iloc[i] > 2]
vpi2_mz4_lambda1_vcm2_arr = [vpi_double_mz4_lambda1_vcm2.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz4_lambda1_vcm2))
                             if vpi_double_mz4_lambda1_vcm2.iloc[i] > 2]

vpi2_mz1_lambda1_vcm2 = np.mean(vpi2_mz1_lambda1_vcm2_arr)
vpi2_mz2_lambda1_vcm2 = np.mean(vpi2_mz2_lambda1_vcm2_arr)
vpi2_mz3_lambda1_vcm2 = np.mean(vpi2_mz3_lambda1_vcm2_arr)
vpi2_mz4_lambda1_vcm2 = np.mean(vpi2_mz4_lambda1_vcm2_arr)

vpi_mz1_lambda1_vcm2 = vpi2_mz1_lambda1_vcm2/2
vpi_mz2_lambda1_vcm2 = vpi2_mz2_lambda1_vcm2/2
vpi_mz3_lambda1_vcm2 = vpi2_mz3_lambda1_vcm2/2
vpi_mz4_lambda1_vcm2 = vpi2_mz4_lambda1_vcm2/2

vpi2_lambda1_vcm2_mean = np.mean([vpi2_mz1_lambda1_vcm2, vpi2_mz2_lambda1_vcm2,
                                  vpi2_mz3_lambda1_vcm2, vpi2_mz4_lambda1_vcm2])

vpi_lambda1_vcm2_mean = vpi2_lambda1_vcm2_mean/2

vpi_double_mz1_lambda2_vcm2 = df_lambda2["MZ1_RF_DIFF_VCM2_2VPI_V"]
vpi_double_mz2_lambda2_vcm2 = df_lambda2["MZ2_RF_DIFF_VCM2_2VPI_V"]
vpi_double_mz3_lambda2_vcm2 = df_lambda2["MZ3_RF_DIFF_VCM2_2VPI_V"]
vpi_double_mz4_lambda2_vcm2 = df_lambda2["MZ4_RF_DIFF_VCM2_2VPI_V"]

vpi2_mz1_lambda2_vcm2_arr = [vpi_double_mz1_lambda2_vcm2.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz1_lambda2_vcm2))
                             if vpi_double_mz1_lambda2_vcm2.iloc[i] > 2]
vpi2_mz2_lambda2_vcm2_arr = [vpi_double_mz2_lambda2_vcm2.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz2_lambda2_vcm2))
                             if vpi_double_mz2_lambda2_vcm2.iloc[i] > 2]
vpi2_mz3_lambda2_vcm2_arr = [vpi_double_mz3_lambda2_vcm2.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz3_lambda2_vcm2))
                             if vpi_double_mz3_lambda2_vcm2.iloc[i] > 2]
vpi2_mz4_lambda2_vcm2_arr = [vpi_double_mz4_lambda2_vcm2.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz4_lambda2_vcm2))
                             if vpi_double_mz4_lambda2_vcm2.iloc[i] > 2]

vpi2_mz1_lambda2_vcm2 = np.mean(vpi2_mz1_lambda2_vcm2_arr)
vpi2_mz2_lambda2_vcm2 = np.mean(vpi2_mz2_lambda2_vcm2_arr)
vpi2_mz3_lambda2_vcm2 = np.mean(vpi2_mz3_lambda2_vcm2_arr)
vpi2_mz4_lambda2_vcm2 = np.mean(vpi2_mz4_lambda2_vcm2_arr)

vpi_mz1_lambda2_vcm2 = vpi2_mz1_lambda2_vcm2/2
vpi_mz2_lambda2_vcm2 = vpi2_mz2_lambda2_vcm2/2
vpi_mz3_lambda2_vcm2 = vpi2_mz3_lambda2_vcm2/2
vpi_mz4_lambda2_vcm2 = vpi2_mz4_lambda2_vcm2/2

vpi2_lambda2_vcm2_mean = np.mean([vpi2_mz1_lambda2_vcm2, vpi2_mz2_lambda2_vcm2,
                                  vpi2_mz3_lambda2_vcm2, vpi2_mz4_lambda2_vcm2])
vpi_lambda2_vcm2_mean = vpi2_lambda2_vcm2_mean/2

x_new2 = np.linspace(1528.5, 1564.0, 15641-15285)
x2 = [1528.5, 1564]
y2 = [vpi2_lambda1_vcm2_mean, vpi2_lambda2_vcm2_mean]
f2 = interpolate.interp1d(x2,y2)
y_new2 = f2(x_new2)

vpi2_lambda_center_vcm2_mean = y_new2[np.where(x_new2 == 1550.0)]
vpi_lambda_center_vcm2_mean = vpi2_lambda_center_vcm2_mean/2

plt.figure()
plt.plot(x_new2, y_new2)
plt.show(block=False)

# build VCM2 Vpi data structure to fill the vcm_lambda_df
vpi_mzi_lambda1_vcm2 = [vpi_mz1_lambda1_vcm2, vpi_mz2_lambda1_vcm2, vpi_mz3_lambda1_vcm2, vpi_mz4_lambda1_vcm2]
vpi_mzi_lambda2_vcm2 = [vpi_mz1_lambda2_vcm2, vpi_mz2_lambda2_vcm2, vpi_mz3_lambda2_vcm2, vpi_mz4_lambda2_vcm2]

for i in np.arange(np.size(mz_list)):
    mz_label = 'Vpi_'+mz_list[i]
    y = [vpi_mzi_lambda1_vcm2[i], vpi_mzi_lambda2_vcm2[i]]
    f = interpolate.interp1d(lambdas, y)
    y_new = f(x_new)
    df_vcm_lambda[mz_label].iloc[vcm2_indexes] = y_new

# VCM3
vpi_double_mz1_lambda1_vcm3 = df_lambda1["MZ1_RF_DIFF_VCM3_2VPI_V"]
vpi_double_mz2_lambda1_vcm3 = df_lambda1["MZ2_RF_DIFF_VCM3_2VPI_V"]
vpi_double_mz3_lambda1_vcm3 = df_lambda1["MZ3_RF_DIFF_VCM3_2VPI_V"]
vpi_double_mz4_lambda1_vcm3 = df_lambda1["MZ4_RF_DIFF_VCM3_2VPI_V"]

vpi2_mz1_lambda1_vcm3_arr = [vpi_double_mz1_lambda1_vcm3.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz1_lambda1_vcm3))
                             if vpi_double_mz1_lambda1_vcm3.iloc[i] > 1]
vpi2_mz2_lambda1_vcm3_arr = [vpi_double_mz2_lambda1_vcm3.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz2_lambda1_vcm3))
                             if vpi_double_mz2_lambda1_vcm3.iloc[i] > 1]
vpi2_mz3_lambda1_vcm3_arr = [vpi_double_mz3_lambda1_vcm3.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz3_lambda1_vcm3))
                             if vpi_double_mz3_lambda1_vcm3.iloc[i] > 1]
vpi2_mz4_lambda1_vcm3_arr = [vpi_double_mz4_lambda1_vcm3.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz4_lambda1_vcm3))
                             if vpi_double_mz4_lambda1_vcm3.iloc[i] > 1]

vpi2_mz1_lambda1_vcm3 = np.mean(vpi2_mz1_lambda1_vcm3_arr)
vpi2_mz2_lambda1_vcm3 = np.mean(vpi2_mz2_lambda1_vcm3_arr)
vpi2_mz3_lambda1_vcm3 = np.mean(vpi2_mz3_lambda1_vcm3_arr)
vpi2_mz4_lambda1_vcm3 = np.mean(vpi2_mz4_lambda1_vcm3_arr)

vpi_mz1_lambda1_vcm3 = np.mean(vpi2_mz1_lambda1_vcm3_arr)/2
vpi_mz2_lambda1_vcm3 = np.mean(vpi2_mz2_lambda1_vcm3_arr)/2
vpi_mz3_lambda1_vcm3 = np.mean(vpi2_mz3_lambda1_vcm3_arr)/2
vpi_mz4_lambda1_vcm3 = np.mean(vpi2_mz4_lambda1_vcm3_arr)/2

vpi2_lambda1_vcm3_mean = np.mean([vpi2_mz1_lambda1_vcm3, vpi2_mz2_lambda1_vcm3,
                                  vpi2_mz3_lambda1_vcm3, vpi2_mz4_lambda1_vcm3])

vpi_lambda1_vcm3_mean = vpi2_lambda1_vcm3_mean/2

vpi_double_mz1_lambda2_vcm3 = df_lambda2["MZ1_RF_DIFF_VCM3_2VPI_V"]
vpi_double_mz2_lambda2_vcm3 = df_lambda2["MZ2_RF_DIFF_VCM3_2VPI_V"]
vpi_double_mz3_lambda2_vcm3 = df_lambda2["MZ3_RF_DIFF_VCM3_2VPI_V"]
vpi_double_mz4_lambda2_vcm3 = df_lambda2["MZ4_RF_DIFF_VCM3_2VPI_V"]

vpi2_mz1_lambda2_vcm3_arr = [vpi_double_mz1_lambda2_vcm3.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz1_lambda2_vcm3))
                             if vpi_double_mz1_lambda2_vcm3.iloc[i] > 1]
vpi2_mz2_lambda2_vcm3_arr = [vpi_double_mz2_lambda2_vcm3.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz2_lambda2_vcm3))
                             if vpi_double_mz2_lambda2_vcm3.iloc[i] > 1]
vpi2_mz3_lambda2_vcm3_arr = [vpi_double_mz3_lambda2_vcm3.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz3_lambda2_vcm3))
                             if vpi_double_mz3_lambda2_vcm3.iloc[i] > 1]
vpi2_mz4_lambda2_vcm3_arr = [vpi_double_mz4_lambda2_vcm3.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz4_lambda2_vcm3))
                             if vpi_double_mz4_lambda2_vcm3.iloc[i] > 1]

vpi2_mz1_lambda2_vcm3 = np.mean(vpi2_mz1_lambda2_vcm3_arr)
vpi2_mz2_lambda2_vcm3 = np.mean(vpi2_mz2_lambda2_vcm3_arr)
vpi2_mz3_lambda2_vcm3 = np.mean(vpi2_mz3_lambda2_vcm3_arr)
vpi2_mz4_lambda2_vcm3 = np.mean(vpi2_mz4_lambda2_vcm3_arr)

vpi_mz1_lambda2_vcm3 = np.mean(vpi2_mz1_lambda2_vcm3_arr)/2
vpi_mz2_lambda2_vcm3 = np.mean(vpi2_mz2_lambda2_vcm3_arr)/2
vpi_mz3_lambda2_vcm3 = np.mean(vpi2_mz3_lambda2_vcm3_arr)/2
vpi_mz4_lambda2_vcm3 = np.mean(vpi2_mz4_lambda2_vcm3_arr)/2

vpi2_lambda2_vcm3_mean = np.mean([vpi2_mz1_lambda2_vcm3, vpi2_mz2_lambda2_vcm3,
                                  vpi2_mz3_lambda2_vcm3, vpi2_mz4_lambda2_vcm3])

vpi_lambda2_vcm3_mean = vpi2_lambda2_vcm3_mean/2

x_new3 = np.linspace(1528.5, 1564.0, 15641-15285)
x3 = [1528.5, 1564]
y3 = [vpi2_lambda1_vcm3_mean, vpi2_lambda2_vcm3_mean]
f3 = interpolate.interp1d(x3,y3)
y_new3 = f3(x_new3)

vpi2_lambda_center_vcm3_mean = y_new3[np.where(x_new3 == 1550.0)]
vpi_lambda_center_vcm3_mean = vpi2_lambda_center_vcm3_mean/2

plt.plot(x_new3, y_new3)
plt.show(block=False)

# build VCM3 Vpi data structure to fill the vcm_lambda_df
vpi_mzi_lambda1_vcm3 = [vpi_mz1_lambda1_vcm3, vpi_mz2_lambda1_vcm3, vpi_mz3_lambda1_vcm3, vpi_mz4_lambda1_vcm3]
vpi_mzi_lambda2_vcm3 = [vpi_mz1_lambda2_vcm3, vpi_mz2_lambda2_vcm3, vpi_mz3_lambda2_vcm3, vpi_mz4_lambda2_vcm3]

for i in np.arange(np.size(mz_list)):
    mz_label = 'Vpi_'+mz_list[i]
    y = [vpi_mzi_lambda1_vcm3[i], vpi_mzi_lambda2_vcm3[i]]
    f = interpolate.interp1d(lambdas, y)
    y_new = f(x_new)
    df_vcm_lambda[mz_label].iloc[vcm3_indexes] = y_new

# VCM4
vpi_double_mz1_lambda1_vcm4 = df_lambda1["MZ1_RF_DIFF_VCM4_2VPI_V"]
vpi_double_mz2_lambda1_vcm4 = df_lambda1["MZ2_RF_DIFF_VCM4_2VPI_V"]
vpi_double_mz3_lambda1_vcm4 = df_lambda1["MZ3_RF_DIFF_VCM4_2VPI_V"]
vpi_double_mz4_lambda1_vcm4 = df_lambda1["MZ4_RF_DIFF_VCM4_2VPI_V"]

vpi2_mz1_lambda1_vcm4_arr = [vpi_double_mz1_lambda1_vcm4.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz1_lambda1_vcm4))
                             if vpi_double_mz1_lambda1_vcm4.iloc[i] > 0]
vpi2_mz2_lambda1_vcm4_arr = [vpi_double_mz2_lambda1_vcm4.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz2_lambda1_vcm4))
                             if vpi_double_mz2_lambda1_vcm4.iloc[i] > 0]
vpi2_mz3_lambda1_vcm4_arr = [vpi_double_mz3_lambda1_vcm4.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz3_lambda1_vcm4))
                             if vpi_double_mz3_lambda1_vcm4.iloc[i] > 0]
vpi2_mz4_lambda1_vcm4_arr = [vpi_double_mz4_lambda1_vcm4.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz4_lambda1_vcm4))
                             if vpi_double_mz4_lambda1_vcm4.iloc[i] > 0]

vpi2_mz1_lambda1_vcm4 = np.mean(vpi2_mz1_lambda1_vcm4_arr)
vpi2_mz2_lambda1_vcm4 = np.mean(vpi2_mz2_lambda1_vcm4_arr)
vpi2_mz3_lambda1_vcm4 = np.mean(vpi2_mz3_lambda1_vcm4_arr)
vpi2_mz4_lambda1_vcm4 = np.mean(vpi2_mz4_lambda1_vcm4_arr)

vpi_mz1_lambda1_vcm4 = np.mean(vpi2_mz1_lambda1_vcm4_arr)/2
vpi_mz2_lambda1_vcm4 = np.mean(vpi2_mz2_lambda1_vcm4_arr)/2
vpi_mz3_lambda1_vcm4 = np.mean(vpi2_mz3_lambda1_vcm4_arr)/2
vpi_mz4_lambda1_vcm4 = np.mean(vpi2_mz4_lambda1_vcm4_arr)/2

vpi2_lambda1_vcm4_mean = np.mean([vpi2_mz1_lambda1_vcm4, vpi2_mz2_lambda1_vcm4,
                                  vpi2_mz3_lambda1_vcm4, vpi2_mz4_lambda1_vcm4])

vpi_lambda1_vcm4_mean = vpi2_lambda1_vcm4_mean/2

vpi_double_mz1_lambda2_vcm4 = df_lambda2["MZ1_RF_DIFF_VCM4_2VPI_V"]
vpi_double_mz2_lambda2_vcm4 = df_lambda2["MZ2_RF_DIFF_VCM4_2VPI_V"]
vpi_double_mz3_lambda2_vcm4 = df_lambda2["MZ3_RF_DIFF_VCM4_2VPI_V"]
vpi_double_mz4_lambda2_vcm4 = df_lambda2["MZ4_RF_DIFF_VCM4_2VPI_V"]

vpi2_mz1_lambda2_vcm4_arr = [vpi_double_mz1_lambda2_vcm4.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz1_lambda2_vcm4))
                             if vpi_double_mz1_lambda2_vcm4.iloc[i] > 0]
vpi2_mz2_lambda2_vcm4_arr = [vpi_double_mz2_lambda2_vcm4.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz2_lambda2_vcm4))
                             if vpi_double_mz2_lambda2_vcm4.iloc[i] > 0]
vpi2_mz3_lambda2_vcm4_arr = [vpi_double_mz3_lambda2_vcm4.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz3_lambda2_vcm4))
                             if vpi_double_mz3_lambda2_vcm4.iloc[i] > 0]
vpi2_mz4_lambda2_vcm4_arr = [vpi_double_mz4_lambda2_vcm4.iloc[i]
                             for i in np.arange(np.size(vpi_double_mz4_lambda2_vcm4))
                             if vpi_double_mz4_lambda2_vcm4.iloc[i] > 0]

vpi2_mz1_lambda2_vcm4 = np.mean(vpi2_mz1_lambda2_vcm4_arr)
vpi2_mz2_lambda2_vcm4 = np.mean(vpi2_mz2_lambda2_vcm4_arr)
vpi2_mz3_lambda2_vcm4 = np.mean(vpi2_mz3_lambda2_vcm4_arr)
vpi2_mz4_lambda2_vcm4 = np.mean(vpi2_mz4_lambda2_vcm4_arr)

vpi_mz1_lambda2_vcm4 = np.mean(vpi2_mz1_lambda2_vcm4_arr)/2
vpi_mz2_lambda2_vcm4 = np.mean(vpi2_mz2_lambda2_vcm4_arr)/2
vpi_mz3_lambda2_vcm4 = np.mean(vpi2_mz3_lambda2_vcm4_arr)/2
vpi_mz4_lambda2_vcm4 = np.mean(vpi2_mz4_lambda2_vcm4_arr)/2

vpi2_lambda2_vcm4_mean = np.mean([vpi2_mz1_lambda2_vcm4, vpi2_mz2_lambda2_vcm4,
                                  vpi2_mz3_lambda2_vcm4, vpi2_mz4_lambda2_vcm4])

vpi_lambda2_vcm4_mean = vpi2_lambda2_vcm4_mean/2

x_new4 = np.linspace(1528.5, 1564.0, 15641-15285)
x4 = [1528.5, 1564]
y4 = [vpi2_lambda1_vcm4_mean, vpi2_lambda2_vcm4_mean]
f4 = interpolate.interp1d(x4,y4)
y_new4 = f4(x_new4)

vpi2_lambda_center_vcm4_mean = y_new4[np.where(x_new4 == 1550.0)]
vpi_lambda_center_vcm4_mean = vpi2_lambda_center_vcm4_mean/2

plt.figure()
plt.plot(x_new4, y_new4)
plt.show(block=False)

# build VCM4 Vpi data structure to fill the vcm_lambda_df
vpi_mzi_lambda1_vcm4 = [vpi_mz1_lambda1_vcm4, vpi_mz2_lambda1_vcm4, vpi_mz3_lambda1_vcm4, vpi_mz4_lambda1_vcm4]
vpi_mzi_lambda2_vcm4 = [vpi_mz1_lambda2_vcm4, vpi_mz2_lambda2_vcm4, vpi_mz3_lambda2_vcm4, vpi_mz4_lambda2_vcm4]

for i in np.arange(np.size(mz_list)):
    mz_label = 'Vpi_'+mz_list[i]
    y = [vpi_mzi_lambda1_vcm4[i], vpi_mzi_lambda2_vcm4[i]]
    f = interpolate.interp1d(lambdas, y)
    y_new = f(x_new)
    df_vcm_lambda[mz_label].iloc[vcm4_indexes] = y_new

"""TODO: FIND THE MEAN FOR EACH CASE, THEN INTERPOLATE ALL LINEARLY AND FIND 2VPI FOR EACH LAMBDA
IN PARTICULAR FOR 1550nm"""
print("2*Vpi:")
print(vpi2_lambda_center_vcm1_mean)
print(vpi2_lambda_center_vcm2_mean)
print(vpi2_lambda_center_vcm3_mean)
print(vpi2_lambda_center_vcm4_mean)
print("Vpi:")
print(vpi_lambda_center_vcm1_mean)
print(vpi_lambda_center_vcm2_mean)
print(vpi_lambda_center_vcm3_mean)
print(vpi_lambda_center_vcm4_mean)

df_vcm_lambda.to_csv("results/vpi_vcm_mzi_lambda.csv", index=False)

plt.figure()
plt.plot(x_new1, y_new1)
plt.plot(x_new2, y_new2)
plt.plot(x_new3, y_new3)
plt.plot(x_new4, y_new4)
plt.show(block=True)