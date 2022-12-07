'''
1.cal the ERA5 eddyz500 in 1959-2021
2.cal the eddyz500 anomaly in NA (no mask+no detrend)
2.save anomaly data
'''

import numpy as np
import xarray as xr
import os
import scipy
from script_for_NAheatdome import cal_func
import pandas as pd

#============================
#1.读取ERA5的z500
path = "/WORK2/zhangx/program/flow_ana/data2/ERA5/z500/"
# 读取该文件夹下的nc文件
nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".nc")]
nc_files.sort() #排序!!!!
print(nc_files)
# 读取nc文件，这里直接将多个文件按照时间维度合并
ds = xr.open_mfdataset(nc_files,combine="by_coords")
print(ds)

#2.计算出eddyz500
z500 = ds.z/9.8  #gpm
print(z500)
z500 = z500.convert_calendar('365_day')
z500_zonalmean = z500.mean("lon")   
eddyz500 = z500.transpose('lon','time','lat') - z500_zonalmean.transpose('time','lat')
eddyz500 = eddyz500.transpose('time','lat','lon')
print(eddyz500)
del(z500)
del(z500_zonalmean)

#3.计算anomaly
eddyz500 = eddyz500.sel(lat=slice(20,80),lon=slice(200,300))
eddyz500 = cal_func.cal_anomaly(eddyz500,'1981-05-01','2010-08-31','False')


#存储数据
year    = np.arange(1959,2022)
nyr     = len(year)
time = list(pd.date_range('1959-05-01','1959-08-31',freq='D'))
for y in year[1:nyr]:
	time_a2 = list(pd.date_range(str(int(y))+'-05-01',str(int(y))+'-08-31',freq='D'))  #1949-06-01:2020-08-31
	time.extend(time_a2)
time  = pd.DatetimeIndex(time, dtype='datetime64[ns]', freq=None)
print(time)
eddyz500['time'] = time
print(eddyz500)

dss = xr.Dataset({'eddyz500_ano':eddyz500})   #DataArray转成DataSet
dss.to_netcdf('/WORK2/zhangx/program/flow_ana/data2/ERA5/ano_eddyz500/ERA5_daily_eddyz500_anomaly_1959-2021.nc', engine='netcdf4')


#2021年的eddy z500 anomaly还需要单独存储
ds2 = xr.Dataset({'eddyz500_ano':eddyz500.sel(time=slice('2021-05-01','2021-08-31'))})   #DataArray转成DataSet
ds2.to_netcdf('/WORK2/zhangx/program/flow_ana/data2/ERA5/ano_eddyz500/ERA5_daily_eddyz500_anomaly_2021.nc', engine='netcdf4')


