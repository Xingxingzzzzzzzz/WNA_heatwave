'''
1.cal the tmax anomaly in 1959-2021(mask+no detrend)
2.save above data
'''

import numpy as np
import scipy
import os
import netCDF4 as nc
import pandas as pd
from scipy import signal
from script_for_NAheatdome import cal_func
import xarray as xr
from fnmatch import fnmatch, fnmatchcase
from csaps import csaps


#============================================
#1.读取ERA5日最高温度数据,并计算anomaly
path = "/WORK2/zhangx/program/flow_ana/data2/ERA5/mx2t/"
# 读取该文件夹下的nc文件
nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".nc")]
print(nc_files)
# 读取nc文件，这里直接将多个文件按照时间维度合并
ds = xr.open_mfdataset(nc_files,combine="by_coords")
print(ds)


##创建时间,这里需要转为datetime64的格式！！
year    = np.arange(1959,2022)
nyr     = len(year)
time = list(pd.date_range('1959-05-01','1959-08-31',freq='D'))
for y in year[1:nyr]:
	time_a2 = list(pd.date_range(str(int(y))+'-05-01',str(int(y))+'-08-31',freq='D'))  #1949-06-01:2020-08-31
	time.extend(time_a2)
time  = pd.DatetimeIndex(time, dtype='datetime64[ns]', freq=None)
print(time)

#----nodetrend----
mx2t     = ds.mx2t-273.15
mx2t     = mx2t.convert_calendar('365_day')
print(mx2t)
#anomaly
tmax_ano = cal_func.cal_anomaly(mx2t,'1981-05-01','2010-08-31','True')
print(tmax_ano)

tmax_ano['time'] = time
print(tmax_ano)

##存储数据
dss = xr.Dataset({'tmx_ano':tmax_ano})   #DataArray转成DataSet
dss.to_netcdf('/WORK2/zhangx/program/flow_ana/data2/ERA5/ano_tmax/ERA5_daily_tmx_anomaly_1959-2021.nc', engine='netcdf4')
print(okk)


