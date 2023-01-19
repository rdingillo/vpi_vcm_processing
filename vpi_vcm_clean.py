import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate


df = pd.read_csv(
    "C:/Users//rocco/OneDrive - Politecnico di Torino/Desktop//Caratterizzazione Vpi-Vcm/Vpi_Dec_22/Vpi_Dec_22_sep2.csv",
    sep=';')

df_lambda1 = df.loc[df["WAVELENGTH_NM"] == 1528.5]
df_lambda2 = df.loc[df["WAVELENGTH_NM"] == 1564]

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

vpi2_lambda1_vcm1_mean = np.mean([vpi2_mz1_lambda1_vcm1, vpi2_mz2_lambda1_vcm1,
                                  vpi2_mz3_lambda1_vcm1, vpi2_mz4_lambda1_vcm1])

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

vpi2_lambda2_vcm1_mean = np.mean([vpi2_mz1_lambda2_vcm1, vpi2_mz2_lambda2_vcm1,
                                  vpi2_mz3_lambda2_vcm1, vpi2_mz4_lambda2_vcm1])

x_new1 = np.linspace(1528.5, 1564.0, 15641-15285)
x1 = [1528.5, 1564]
y1 = [vpi2_lambda1_vcm1_mean, vpi2_lambda2_vcm1_mean]
f1 = interpolate.interp1d(x1,y1)
y_new1 = f1(x_new1)

vpi2_lambda_center_vcm1_mean = y_new1[np.where(x_new1 == 1550.0)]

plt.plot(x_new1, y_new1)
plt.show(block=False)

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

vpi2_lambda1_vcm2_mean = np.mean([vpi2_mz1_lambda1_vcm2, vpi2_mz2_lambda1_vcm2,
                                  vpi2_mz3_lambda1_vcm2, vpi2_mz4_lambda1_vcm2])

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

vpi2_lambda2_vcm2_mean = np.mean([vpi2_mz1_lambda2_vcm2, vpi2_mz2_lambda2_vcm2,
                                  vpi2_mz3_lambda2_vcm2, vpi2_mz4_lambda2_vcm2])

x_new2 = np.linspace(1528.5, 1564.0, 15641-15285)
x2 = [1528.5, 1564]
y2 = [vpi2_lambda1_vcm2_mean, vpi2_lambda2_vcm2_mean]
f2 = interpolate.interp1d(x2,y2)
y_new2 = f2(x_new2)

vpi2_lambda_center_vcm2_mean = y_new2[np.where(x_new2 == 1550.0)]

plt.figure()
plt.plot(x_new2, y_new2)
plt.show(block=False)

# plt.figure()
# plt.plot(x_new1, y_new1)
# plt.plot(x_new2, y_new2)
# plt.show(block=True)

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

vpi2_lambda1_vcm3_mean = np.mean([vpi2_mz1_lambda1_vcm3, vpi2_mz2_lambda1_vcm3,
                                  vpi2_mz3_lambda1_vcm3, vpi2_mz4_lambda1_vcm3])

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

vpi2_lambda2_vcm3_mean = np.mean([vpi2_mz1_lambda2_vcm3, vpi2_mz2_lambda2_vcm3,
                                  vpi2_mz3_lambda2_vcm3, vpi2_mz4_lambda2_vcm3])

x_new3 = np.linspace(1528.5, 1564.0, 15641-15285)
x3 = [1528.5, 1564]
y3 = [vpi2_lambda1_vcm3_mean, vpi2_lambda2_vcm3_mean]
f3 = interpolate.interp1d(x3,y3)
y_new3 = f3(x_new3)

vpi2_lambda_center_vcm3_mean = y_new3[np.where(x_new3 == 1550.0)]

plt.plot(x_new3, y_new3)
plt.show(block=False)

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

vpi2_lambda1_vcm4_mean = np.mean([vpi2_mz1_lambda1_vcm4, vpi2_mz2_lambda1_vcm4,
                                  vpi2_mz3_lambda1_vcm4, vpi2_mz4_lambda1_vcm4])

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

vpi2_lambda2_vcm4_mean = np.mean([vpi2_mz1_lambda2_vcm4, vpi2_mz2_lambda2_vcm4,
                                  vpi2_mz3_lambda2_vcm4, vpi2_mz4_lambda2_vcm4])

x_new4 = np.linspace(1528.5, 1564.0, 15641-15285)
x4 = [1528.5, 1564]
y4 = [vpi2_lambda1_vcm4_mean, vpi2_lambda2_vcm4_mean]
f4 = interpolate.interp1d(x4,y4)
y_new4 = f4(x_new4)

vpi2_lambda_center_vcm4_mean = y_new4[np.where(x_new4 == 1550.0)]

plt.figure()
plt.plot(x_new4, y_new4)
plt.show(block=False)

plt.figure()
plt.plot(x_new1, y_new1)
plt.plot(x_new2, y_new2)
plt.plot(x_new3, y_new3)
plt.plot(x_new4, y_new4)
plt.show(block=True)
"""TODO: FIND THE MEAN FOR EACH CASE, THEN INTERPOLATE ALL LINEARLY AND FIND 2VPI FOR EACH LAMBDA
IN PARTICULAR FOR 1550nm"""
print(vpi2_lambda_center_vcm1_mean)
print(vpi2_lambda_center_vcm2_mean)
print(vpi2_lambda_center_vcm3_mean)
print(vpi2_lambda_center_vcm4_mean)


