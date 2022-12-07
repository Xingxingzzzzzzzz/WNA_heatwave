
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import sys
sys.path.append('/WORK2/zhangx/program/def')
from script_for_NAheatdome import cal_func
import random
import netCDF4 as nc
import scipy
from fnmatch import fnmatch, fnmatchcase
import calendar
from scipy import signal
import scipy.stats as stats


#=====================================
#ERA5
#1.读取数据
ds1      = xr.open_dataset('/WORK2/zhangx/program/flow_ana/data2/ERA5/ano_tmax/ERA5_daily_tmx_anomaly_1959-2021.nc')
tmax_ano = ds1.tmx_ano
print(tmax_ano)
lat      = tmax_ano.coords['lat']
lon      = tmax_ano.coords['lon']

ds2          = xr.open_dataset('/WORK2/zhangx/program/flow_ana/data2/ERA5/ano_eddyz500/ERA5_daily_eddyz500_anomaly_1959-2021.nc')
eddyz500_ano = ds2.eddyz500_ano
print(eddyz500_ano)

#对daily的数据区域平均，然后去趋势
weights     = np.cos(np.deg2rad(tmax_ano.sel(lat=slice(40,65),lon=slice(235,255)).lat))
tmax_ano_ts = tmax_ano.sel(lat=slice(40,65),lon=slice(235,255)).weighted(weights).mean(("lat","lon")) #原时间序列，未去趋势
#detrend
tmax_ano_ts2 = cal_func.detrend_fit(tmax_ano_ts) #这里是对区域平均的序列去趋势！
tmax_ano_21  = tmax_ano_ts2.sel(time=slice('2021-06-27','2021-07-03')).mean('time')
print(f'2021 the tmax anomaly is {tmax_ano_21.data:.2f}') #

#*****************whole time 1959-2020*******************
#-------------读取选取的环流类似的日期----------------
#filepath = '/WORK2/zhangx/program/flow_ana/data2/ANA/'
#data = pd.read_csv(filepath+'analogue_date/ERA5/ana_eddyz500_ano_500_rms_NA_NW_sim_2021-06-27_2021-07-03_base_1959-05-01_2020-08-31_235.0_255.0_40.0_65.0_1_30.txt',\
#       sep='\s+',header=None)
#print(data)
#day  = np.array(data)
#day  = day[1:8,1:21]
#
###================基于环流相似取样=============
#nn = 1000 #取样次数
#v  = cal_func(tmax_ano_ts2)      #去趋势的温度序列
#ta_ana,eddyz500ano_ana,ta_ana2 = v.ana_random_detrend(day,7,eddyz500_ano,nn,tmax_ano=tmax_ano)  
##ta_ana(1000,) eddyz500ano_ana(1000,ny,nx),ta_ana2(1000,ny,nx)
#
##============赋属性并存储1000次取样的数据==============
##(20:80N,160W:60W)
#ds4 = xr.Dataset({'ta':(('n'),ta_ana),'tmax_ano':(('n','lat','lon'),ta_ana2),'eddyz500_ano':(('n','lat','lon'),eddyz500ano_ana)},\
#	             coords={'n':np.arange(1,nn+1),'lat':lat,'lon':lon},\
#	             attrs=dict(description="tmax anomaly (pattern) & eddy z500 anomaly (pattern) base on flow analogue")) 
#outputfile4=filepath+'ERA5/1ERA5_ana_tmax_eddyz500_ano_pattern_1959-2020.nc'
#ds4.to_netcdf(outputfile4, engine='netcdf4')

#=================随意取样，控制实验==================
#detrend
#vv        = cal_func(tmax_ano_ts2)
#ta_random = vv.random_nn(1959,2020,7,nn,'05-28','08-02','05-31','07-30') #(1000)
#
####==============存储数据==============
#ds2 = xr.Dataset({'ta_r':(('n'),ta_random)},\
#                    coords={'n':np.arange(1,nn+1)},\
#                    attrs=dict(description="tmax anomaly from lots of randomly picked"))
#outputfile2=filepath+'ERA5/1ERA5_ana_tmax_ano_randomly_1959-2020.nc'
#ds2.to_netcdf(outputfile2, engine='netcdf4')

#ta_m_uchronic = np.nanmean(ta_ana)    
#print(f'ERA5 mean={ta_m_uchronic:.2f}')
#
#ta_median = np.nanmedian(ta_ana)      
#print(f'median={ta_median:.2f}')
#
#r  = ta_median/tmax_ano_21*100   
#print(f'{r.data:.2f}%')
#
##ERA5 mean=3.67
##median=3.67
##54.64%
#
#
#print(okk)

##*****************前期，1959-1990*********************
#-------------读取选取的环流类似的日期----------------
filepath = '/WORK2/zhangx/program/flow_ana/data2/ANA/'
data = pd.read_csv(filepath+'analogue_date/ERA5/ana_eddyz500_ano_500_rms_NA_NW_sim_2021-06-27_2021-07-03_base_1959-05-01_1990-08-31_235.0_255.0_40.0_65.0_1_30.txt',\
       sep='\s+',header=None)
print(data)
day  = np.array(data)
day  = day[1:8,1:21]
print(day)

##================基于环流相似取样=============
nn = 1000 #取样次数
sel_ts2 = dict(time=slice('1959-05-01','1990-08-31'))
v  = cal_func(tmax_ano_ts2.sel(**sel_ts2))
ta_ana,eddyz500ano_ana,ta_ana_nod = v.ana_random_detrend(day,7,eddyz500_ano.sel(**sel_ts2),nn,tmax_ano_ts.sel(**sel_ts2))

#============赋属性并存储1000次取样的空间数据==============
#(20:80N,160W:60W)
#detrend
#ds4 = xr.Dataset({'ta':(('n'),ta_ana),'ta_nd':(('n'),ta_ana_nod),'eddyz500_ano':(('n','lat','lon'),eddyz500ano_ana)},\
#	             coords={'n':np.arange(1,nn+1),'lat':lat,'lon':lon},\
#	             attrs=dict(description="tmax anomaly & eddy z500 anomaly(pattern) base on flow analogue")) 
#outputfile4=filepath+'ERA5/2ERA5_ana_tmax_eddyz500_ano_pattern_1959-1990.nc'
#ds4.to_netcdf(outputfile4, engine='netcdf4')
#
#
##=================随意取样，控制实验==================
#vv         = cal_func(tmax_ano_ts2)
#ta_random  = vv.random_nn(1959,1990,7,nn,'05-28','08-02','05-31','07-30') #(1000) , detrend
#vv         = cal_func(tmax_ano_ts)
#ta_random2 = vv.random_nn(1959,1990,7,nn,'05-28','08-02','05-31','07-30')  #nodetrend
#
#
####==============存储数据==============
#ds2 = xr.Dataset({'ta_r':(('n'),ta_random),'ta_r_nd':(('n'),ta_random2)},\
#                    coords={'n':np.arange(1,nn+1)},\
#                    attrs=dict(description="tmax anomaly from lots of randomly picked"))
#outputfile2=filepath+'ERA5/2ERA5_ana_tmax_ano_randomly_1959-1990.nc'
#ds2.to_netcdf(outputfile2, engine='netcdf4')
#
#----计算环流的贡献-----
ta_ana1        = ta_ana
ta_m_uchronic  = np.nanmean(ta_ana1)    
print(f'ERA5 mean={ta_m_uchronic:.2f}')

ta_median = np.nanmedian(ta_ana1)      
print(f'median={ta_median:.2f}')

r  = ta_median/tmax_ano_21*100   
print(f'{r.data:.2f}%')

##*****************后期，1991-2020*********************
##-------------读取选取的环流类似的日期----------------
filepath = '/WORK2/zhangx/program/flow_ana/data2/ANA/'
data = pd.read_csv(filepath+'analogue_date/ERA5/ana_eddyz500_ano_500_rms_NA_NW_sim_2021-06-27_2021-07-03_base_1991-05-01_2020-08-31_235.0_255.0_40.0_65.0_1_30.txt',\
       sep='\s+',header=None)
print(data)
day  = np.array(data)
day  = day[1:8,1:21]
print(day)

##================基于环流相似取样=============
nn = 1000 #取样次数
sel_ts2 = dict(time=slice('1991-05-01','2020-08-31'))
v  = cal_func(tmax_ano_ts2.sel(**sel_ts2))
ta_ana,eddyz500ano_ana,ta_ana_nod = v.ana_random_detrend(day,7,eddyz500_ano.sel(**sel_ts2),nn,tmax_ano_ts.sel(**sel_ts2))  

#============赋属性并存储1000次取样的空间数据==============
#(20:80N,160W:60W)
#detrend
#ds4 = xr.Dataset({'ta':(('n'),ta_ana),'ta_nd':(('n'),ta_ana_nod),'eddyz500_ano':(('n','lat','lon'),eddyz500ano_ana)},\
#	             coords={'n':np.arange(1,nn+1),'lat':lat,'lon':lon},\
#	             attrs=dict(description="tmax anomaly & eddy z500 anomaly base on flow analogue")) 
#outputfile4=filepath+'ERA5/3ERA5_ana_tmax_eddyz500_ano_pattern_1991-2020.nc'
#ds4.to_netcdf(outputfile4, engine='netcdf4')
#
###=================随意取样，控制实验==================
#vv        = cal_func(tmax_ano_ts2)
#ta_random = vv.random_nn(1991,2020,7,nn,'05-28','08-02','05-31','07-30') 
#vv        = cal_func(tmax_ano_ts)
#ta_random2= vv.random_nn(1991,2020,7,nn,'05-28','08-02','05-31','07-30') 
#
#####==============存储数据==============
#ds2 = xr.Dataset({'ta_r':(('n'),ta_random),'ta_r_nd':(('n'),ta_random2)},\
#                    coords={'n':np.arange(1,nn+1)},\
#                    attrs=dict(description="tmax anomaly from lots of randomly picked"))
#outputfile2=filepath+'ERA5/3ERA5_ana_tmax_ano_randomly_1991-2020.nc'
#ds2.to_netcdf(outputfile2, engine='netcdf4')
#
#---计算环流贡献---
ta_m_uchronic1 = np.nanmean(ta_ana1)    
print(f'ERA5 mean={ta_m_uchronic1:.2f}')

ta_median1 = np.nanmedian(ta_ana1)      
print(f'median={ta_median1:.2f}')

r  = ta_median1/tmax_ano_21*100   
print(f'{r.data:.2f}%')


ta_ana2       = ta_ana
ta_m_uchronic = np.nanmean(ta_ana2)    
print(f'ERA5 mean={ta_m_uchronic:.2f}')

ta_median = np.nanmedian(ta_ana2)      
print(f'median={ta_median:.2f}')

r  = ta_median/tmax_ano_21*100   
print(f'{r.data:.2f}%')

stat_val, p_val = stats.ttest_ind(ta_ana1, ta_ana2, equal_var=False)
print ('ERA5:Two-sample t-statistic D = %6.3f, p-value = %6.4f' % (stat_val, p_val))

print(okk)

#ERA5 mean=3.49
#median=3.49
#52.03%
#ERA5 mean=3.56
#median=3.55
#52.91%
#
#ERA5:Two-sample t-statistic D = -3.130, p-value = 0.0018