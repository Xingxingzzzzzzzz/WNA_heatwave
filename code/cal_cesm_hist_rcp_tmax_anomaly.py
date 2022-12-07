'''
1.cal the CESM tmax anomaly in 1959-2100(mask+no detrend) 以及picontrol的高温数据
2.save above data
'''

import numpy as np
import xarray as xr
import os
import scipy
from script_for_NAheatdome import cal_func
import netCDF4 as nc
from scipy import signal
import pandas as pd
from csaps import csaps


#!!!!!!!!!!!!!!!!!CESM hist+RCP8.5!!!!!!!!!!!!!!!!!!!!!!
#============================
#1.读取多个成员的hist的高温
path = "/WORK2/zhangx/program/flow_ana/data2/CESM/TREFHTMX/hist/"
# 读取该文件夹下的nc文件
nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".nc")]
nc_files.sort() #排序!!!!
print(nc_files)
# 读取nc文件，这里直接将多个文件按照时间维度合并
ds1 = xr.open_mfdataset(nc_files,combine="nested",concat_dim="member").assign_coords(member = np.arange(1,41)) #40member
print(ds1)


#========================
#2.读取RCP85
path2 = "/WORK2/zhangx/program/flow_ana/data2/CESM/TREFHTMX/rcp85/"
# 读取该文件夹下的nc文件
nc_files = [os.path.join(path2, f) for f in os.listdir(path2) if f.endswith(".nc")]
nc_files.sort() #排序!!!!
print(nc_files)
# 读取nc文件，这里直接将多个文件按照时间维度合并
ds2 = xr.open_mfdataset(nc_files,combine="nested",concat_dim="member").assign_coords(member = np.arange(1,41)) #40member
print(ds2)

#拼接hist+rcp85
ds = xr.concat([ds1,ds2],dim='time')
print(ds)
del(ds1)
del(ds2)

#===========================================================
#计算anomaly，每个模式单独保存tmax_anomaly的数据
for i in range(1,41):
	tmax_ano = cal_func.cal_anomaly(ds.TREFHTMX[i-1]-273.15,'1981-05-01','2010-08-31','True')
	time = tmax_ano.indexes['time'].to_datetimeindex() #object转为datetime64格式
	tmax_ano['time'] = time
	#存储数据
	dss = xr.Dataset({'tmx_ano':tmax_ano})   #DataArray转成DataSet
	dss.to_netcdf('/WORK2/zhangx/program/flow_ana/data2/CESM/ano_tmax/hist/cesm_hist_rcp85_tmx_ano_'+str(i).zfill(3)+'.nc', engine='netcdf4')


#------------------------------------------------------
#----提取出历史的气候态，用于计算picontrol的距平----
tmax = ds.TREFHTMX.sel(time=slice('1959-05-01','2021-08-31'))-273.15
tmax = tmax.sel(lat=slice(20,80),lon=slice(200,300))

#计算气候态
sel_ts    = dict(time=slice('1981-05-01','2010-08-31'))   #气候态的时间段,比如1981-2010
tmax_clim = (tmax.sel(**sel_ts).groupby("time.dayofyear").mean("time")).mean("member") #MME的气候态(no-detrend)
print(tmax_clim) #(123,ny,nx)

del(tmax)
#print(okk)

#!!!!!!!!!!!!!!!!!!CESM picontrol!!!!!!!!!!!!!!!!!!!!!!!
#注意！picontrol要减去历史模拟MME的气候态
#==============================================
#1.读取高温数据
path = "/WORK2/zhangx/program/flow_ana/data2/CESM/TREFHTMX/pictrl/"
ds   = xr.open_dataset(path+'b.e11.B1850C5CN.f09_g16.005.cam.h1.TREFHTMX.04020501-20210831.nc') 
print(ds)

#2.计算距平
tmax      = ds.TREFHTMX-273.15
tmax      = tmax.sel(lat=slice(20,80),lon=slice(200,300))
tmax_ano  = (tmax.groupby("time.dayofyear")-tmax_clim).drop("dayofyear") #求完距平后有多余的维度信息，可以drop去掉
print(tmax_ano)

#mask海洋的温度数据
ff3       = nc.Dataset('/WORK2/zhangx/data/ERSST/v5/ersst.v5.1900.2019.1x1.nc')       
sst_mask  = ff3['sst'][0,0,110:171,200:301]
tmax_ano  = xr.where(sst_mask.mask==True,tmax_ano,np.nan)


#存储数据
dss = xr.Dataset({'tmx_ano':tmax_ano})   #DataArray转成DataSet
dss.to_netcdf('/WORK2/zhangx/program/flow_ana/data2/CESM/ano_tmax/pictrl/cesm_picontrol_tmx_ano_04020501-20210831.nc', engine='netcdf4')

