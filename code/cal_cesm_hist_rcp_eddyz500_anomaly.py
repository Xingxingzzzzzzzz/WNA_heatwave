'''
1.cal the CESM eddyz500 in 1959-2100
2.cal the eddyz500 anomaly in NA (no mask+no detrend) and picontrol 
2.save anomaly data
'''

import numpy as np
import xarray as xr
import os
import scipy
from script_for_NAheatdome import cal_func
import pandas as pd
import netCDF4 as nc




#!!!!!!!!!!!!!!!!!!!!CESM hist_RCP8.5!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#============================
#1.读取多个成员的hist的z500
path = "/WORK2/zhangx/program/flow_ana/data2/CESM/Z500/hist/"
# 读取该文件夹下的nc文件
nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".nc")]
nc_files.sort() #排序!!!!
print(nc_files)
# 读取nc文件，这里直接将多个文件按照时间维度合并
ds1 = xr.open_mfdataset(nc_files,combine="nested",concat_dim="member").assign_coords(member = np.arange(1,41)) #40member
print(ds1)


#========================
#2.读取RCP85
path2 = "/WORK2/zhangx/program/flow_ana/data2/CESM/Z500/rcp85/"
# 读取该文件夹下的nc文件
nc_files = [os.path.join(path2, f) for f in os.listdir(path2) if f.endswith(".nc")]
nc_files.sort() #排序!!!
print(nc_files)
# 读取nc文件，这里直接将多个文件按照时间维度合并
ds2 = xr.open_mfdataset(nc_files,combine="nested",concat_dim="member").assign_coords(member = np.arange(1,41)) #40member
print(ds2)

#拼接hist+rcp85
ds = xr.concat([ds1,ds2],dim='time')
print(ds)
del(ds1)
del(ds2)

#==========================================
#3.计算eddyz500,扣除纬向平均
z500 = ds.Z500
print(z500)
z500_zonalmean = z500.mean("lon")   
print(z500_zonalmean)

eddyz500 = z500.transpose('lon','member','time','lat') - z500_zonalmean.transpose('member','time','lat')
print(eddyz500)
eddyz500 = eddyz500.transpose('member','time','lat','lon')
print(eddyz500)
del(z500)
del(z500_zonalmean)

##===========================================================
##计算anomaly，每个模式单独保存eddyz500_anomaly的数据
#for i in range(1,41):
#	eddyz500_ano = cal_func.cal_anomaly(eddyz500[i-1],'1981-05-01','2010-08-31','False')
#	time = eddyz500_ano.indexes['time'].to_datetimeindex() #object转为datetime64格式
#	eddyz500_ano['time'] = time
#	#存储数据
#	dss = xr.Dataset({'eddyz500_ano':eddyz500_ano})   #DataArray转成DataSet
#	dss.to_netcdf('/WORK2/zhangx/program/flow_ana/data2/CESM/ano_eddyz500/hist/cesm_hist_rcp85_eddyz500_ano_'+str(i).zfill(3)+'.nc', engine='netcdf4')
#

eddyz500 = eddyz500.sel(lat=slice(20,80),lon=slice(200,300))
#计算气候态
sel_ts        = dict(time=slice('1981-05-01','2010-08-31'))   #气候态的时间段,比如1981-2010
eddyz500_clim = (eddyz500.sel(**sel_ts).groupby("time.dayofyear").mean("time")).mean("member") #MME的气候态
print(eddyz500_clim)
del(eddyz500)

#!!!!!!!!!!!!!!!!!!CESM picontrol!!!!!!!!!!!!!!!!!!!!!!!
#注意！picontrol要减去历史模拟MME的气候态
#==============================================
#1.读取z500数据
path = "/WORK2/zhangx/program/flow_ana/data2/CESM/Z500/pictrl/"
# 读取该文件夹下的nc文件
nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".nc")]
nc_files.sort() #排序!!!
print(nc_files)
# 读取nc文件，这里直接将多个文件按照时间维度合并
ds = xr.open_mfdataset(nc_files,combine="by_coords") 
print(ds)

#2.计算出eddyz500
z500 = ds.Z500
z500_zonalmean = z500.mean("lon")   
eddyz500 = z500.transpose('lon','time','lat') - z500_zonalmean.transpose('time','lat')
eddyz500 = eddyz500.transpose('time','lat','lon')
print(eddyz500)
del(z500)
del(z500_zonalmean)

#3.计算anomaly,减去hist的气候态
eddyz500 = eddyz500.sel(lat=slice(20,80),lon=slice(200,300))
eddyz500_ano  = (eddyz500.groupby("time.dayofyear")-eddyz500_clim).drop("dayofyear") #求完距平后有多余的维度信息，可以drop去掉
print(eddyz500_ano)

#年份加1000，方便flow analogue的计算
#year2 = np.arange(1402,3022,1)
#time  = list(pd.period_range('1402-05-01', '1402-08-31', freq='D'))
#for y in year2[1:1620]:
#	time2 = list(pd.period_range(str(int(y))+'-05-01',str(int(y))+'-08-31',freq='D'))  #1949-06-01:2020-08-31
#	time.extend(time2)
#time  = pd.PeriodIndex(time, dtype='period[D]', freq=None)
#print(time)

#修改时间402-999 前面补1
import cftime
time    = cftime.num2date(np.arange(1620*365),  units='days since 1402-01-01',calendar='365_day')
time2   = time.reshape((1620,365))[:,120:120+123].reshape((1620*123))
print(time2)


#time2  = cftime.num2date(np.arange(1620*365-1),  units='days since 1402-01-02',calendar='365_day')
#print(time2)
eddyz500_ano = xr.DataArray(np.array(eddyz500_ano),\
		       	 dims=('time','lat','lon'),\
		        coords={'time':time2,'lat':eddyz500.coords['lat'],'lon':eddyz500.coords['lon']})
#nyr = len(year2)
print(eddyz500_ano)


#存储数据
dss = xr.Dataset({'eddyz500_ano':eddyz500_ano})   #DataArray转成DataSet
dss.to_netcdf('/WORK2/zhangx/program/flow_ana/data2/CESM/ano_eddyz500/pictrl/cesm_picontrol_eddyz500_ano_14020501-30210831.nc', engine='netcdf4')



