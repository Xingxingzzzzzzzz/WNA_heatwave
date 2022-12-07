'''
根据ERA5的资料给出此次事件的观测特征，包括温度，环流，txx7序列
并与模式的过去千年的温度作比较
'''

import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
sys.path.append('/WORK2/zhangx/program/def')
from plot_func import contour_map
from plot_func import plot_filled
from script_for_NAheatdome import cal_func
import os
from fnmatch import fnmatch, fnmatchcase
from matplotlib.patches import Polygon
import netCDF4 as nc
import scipy
import cmaps
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde
from scipy.stats import genextreme as gev, kstest
import openturns as ot
from scipy import stats
plt.rc('font',family='Arial')

#===================读取z500===================
#2021年6.27-7.3的异常高度场
#同一个变量不同时间
path = "/WORK2/zhangx/program/flow_ana/data2/ERA5/z500/"
# 读取该文件夹下的nc文件
nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".nc")]
print(nc_files)
# 读取nc文件，这里直接将多个文件按照时间维度合并
ds = xr.open_mfdataset(nc_files,combine="by_coords")
print(ds)

z500   = ds.z.sel(time=slice("1981-06-01","2010-07-31"),lat=slice(0,90),lon=slice(180,360))
z500   = z500.isel(time=z500.time.dt.month.isin([6,7]))/9.8 #只选取6 7月份的数据
print(z500)
#z500 = z500.convert_calendar('365_day')
##日的气候态
#z500_clim = z500.groupby("time.dayofyear").mean("time")
#print(z500_clim)
z500_clim = z500.data.reshape((30,61,z500.data.shape[-2],z500.data.shape[-1]))
z500_clim = np.nanmean(z500_clim,axis=0)

z500_21   = ds.z.sel(time=slice("2021-06-01","2021-07-31"),lat=slice(0,90),lon=slice(180,360))/9.8
lon  = z500_21.coords['lon']
lat  = z500_21.coords['lat']
print(z500_21)

z500_ano  = z500_21 - z500_clim 
print(z500_ano)
#极端高温7天的环流平均
z500_ano  =  z500_ano.sel(time=slice("2021-06-27","2021-07-03")).mean('time')
del(z500)
del(z500_21)


#=======================================
#2021年 6 7 8月的温度异常序列
ds1      = xr.open_dataset('/WORK2/zhangx/program/flow_ana/data2/ERA5/ano_tmax/ERA5_daily_tmx_anomaly_1959-2021.nc')
tmax_ano = ds1.tmx_ano.sel(time=slice('2021-06-01','2021-08-31'))
print(tmax_ano)

#北美西部高温地区做区域平均
weights     = np.cos(np.deg2rad(tmax_ano.sel(lat=slice(40,65),lon=slice(235,255)).lat))
tmax_ano_ts = tmax_ano.sel(lat=slice(40,65),lon=slice(235,255)).weighted(weights).mean(("lat","lon")) #时间序列
print(tmax_ano_ts.loc['2021-06-27':'2021-07-03'].data)
print(np.nanmean(tmax_ano_ts.loc['2021-06-29':'2021-07-01']))
print(np.nanmean(tmax_ano_ts.loc['2021-06-28':'2021-07-02']))

#6.26-7.3
#[ 6.12283118  7.30771903  8.97580167 10.12833166  8.9141852   6.32948429
#  5.52123619]


#==============================================
#ERA5 txx7的序列
ds2       = xr.open_dataset('/WORK2/zhangx/program/flow_ana/data2/ERA5/ERA5_txx7_eddyz500_ano.nc')
txx7_era5 = ds2.txx7
year      = ds2.year
print(txx7_era5)


#计算txx7-ERA5的异常
txx7_era5_std   = txx7_era5.sel(year=slice(1981,2010)).std('year')    #1981-2010年
txx7_era5_clima = txx7_era5.sel(year=slice(1981,2010)).mean('year')
sigma   = (txx7_era5 - txx7_era5_clima)/txx7_era5_std
print(f'sigma is {sigma[-1].data:.2f}')
print(f'mean is {txx7_era5_clima.data:.2f}')  
print(f'std is {txx7_era5_std.data:.2f}')      
print(f'mean+4std:{txx7_era5_clima.data+4*txx7_era5_std.data:.2f}')
print(f'mean+5std:{txx7_era5_clima.data+5*txx7_era5_std.data:.2f}')
print(f'2021 heatwave is {txx7_era5[-1].data:.2f}') 


#sigma is 4.96
#mean is 3.17
#std is 0.90
#mean+4std:6.76
#mean+5std:7.65
#2021 heatwave is 7.61

print(okk)
#================================================
#过去千年的txx7
f2 = xr.open_dataset('/WORK2/zhangx/program/flow_ana/data2/CESM/lastM/cesm1-lastM_txx7_corrected.nc')
txx7 = np.array(f2['txx7'])
print(f2['txx7'])

year2 = np.arange(850,2006)
nyr2  = len(year2)
nm    = 12
x1,y1,p1 = cal_func(txx7.reshape((12*nyr2))).gevv(1)
cal_func(txx7.reshape((12*nyr2))).gevv2()

#-----------------画图-------------
fig1  = plt.figure(figsize=(20,15)) 

proj  = ccrs.PlateCarree()
leftlon, rightlon, lowerlat, upperlat = (-180,0, 0, 90)        
img_extent = [leftlon, rightlon, lowerlat, upperlat]

clevs1 = np.arange(-240,270,30)
#clevs2 = np.arange(5320,6040,80)
#----2021温度序列-----
f_ax1 = fig1.add_axes([0.1,0.6,0.45,0.3])
f_ax1.set_title('(a) 2021 summer tmax anomaly',loc='left',fontsize=25)
f_ax1.set_ylabel('Tmax anomaly ($^\circ$C)',fontsize=25)
#xy_line(f_ax2,year,txx7_era5,1950,2020,10,0)
t     = pd.date_range('2021-06-01','2021-08-31',freq='D')
t2    = pd.DatetimeIndex(t, dtype='datetime64[ns]', freq=None) 
print(t2)
#f_ax2.set_xlim(601,630)
#f_ax2.set_xticks([605,610,615,620,625,630])
f_ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))  #設置x軸主刻度顯示格式（日期）
f_ax1.set_xticks(pd.date_range('2021-06-01','2021-08-31',freq='10D'))
f_ax1.axhline(0,linestyle='-',color='k',linewidth=1)		#设定水平参考线，以及线型，
#f_ax1.axhline(tmax_ano2[-1,2],linestyle='-',color='grey')
plt.xticks(rotation=45)
f_ax1.plot(t2[0:27],tmax_ano_ts[0:27],color='k')
f_ax1.plot(t2[26:33],tmax_ano_ts[26:33],color='red')
f_ax1.plot(t2[32:92],tmax_ano_ts[32:92],color='k') 
f_ax1.tick_params(labelsize=25)
f_ax1.set_ylim(-5,12)
f_ax1.set_yticks(np.arange(-4,16,4))

#xminorLocator   = MultipleLocator(2)
#f_ax1.xaxis.set_minor_locator(xminorLocator)
yminorLocator   = MultipleLocator(1)
f_ax1.yaxis.set_minor_locator(yminorLocator)

#--------历史txx7------
f_ax2  = fig1.add_axes([0.62,0.6,0.45,0.3])
f_ax2.set_title('(b) Summer TXx7',loc='left',fontsize=25)
f_ax2.set_ylabel('Tmax anomaly ($^\circ$C)',fontsize=25)
#xy_line(f_ax2,year,txx7_era5,1950,2020,10,0)
f_ax2.set_xlim(1958,2023)
f_ax2.set_xticks(np.arange(1960,2030,10))
f_ax2.axhline(txx7_era5[-1],linestyle='--',color='red',linewidth=1.5)		#设定水平参考线，以及线型，
f_ax2.plot(year,txx7_era5,color='k') 
f_ax2.plot(2021,txx7_era5[-1],color='r',marker='*',markersize=10)
#f_ax2.axhline(0,linestyle='-',color='k',linewidth=1)    #设定水平参考线，以及线型，
f_ax2.tick_params(labelsize=25)
#f_ax2.set_ylim(-3,6)
#f_ax2.set_yticks(np.arange(-2,8,2))
f_ax2.set_ylim(-2,9)
f_ax2.set_yticks(np.arange(-2,10,2))
xminorLocator   = MultipleLocator(5)
f_ax2.xaxis.set_minor_locator(xminorLocator)
yminorLocator   = MultipleLocator(1)
f_ax2.yaxis.set_minor_locator(yminorLocator)

#---------过去1000年txx7------
f_ax3 = fig1.add_axes([0.62,0.18,0.45,0.3])
f_ax3.set_title('(d) PDF of TXx7',loc='left',fontsize=25)
f_ax3.set_title('CESM1-LME',loc='right',fontsize=25)
f_ax3.set_xlabel('TXx7 ($^\circ$C)',fontsize=25)
f_ax3.set_ylabel('Probability (%)',fontsize=25)

f_ax3.axvline(txx7_era5[-1],linestyle='--',color='tab:red')
#画最冷最热的竖线
color_list = plt.cm.RdBu_r(np.linspace(0, 1, nm*nyr2))
txx7_2 = sorted(txx7.reshape((nm*nyr2)))
f_ax3.axvline(txx7_2[0],linestyle='-',color=color_list[0],alpha=0.6,linewidth=1)
for i in range(1,nm*nyr2):
  if round(txx7_2[i],2)==np.round(txx7_2[i-1],2):
    continue
  else:
    f_ax3.axvline(txx7_2[i],linestyle='-',color=color_list[i],alpha=0.6,linewidth=1)

n_bins  = 30
#-----bar-------
f_ax3.hist(txx7.reshape((nm*nyr2)), n_bins, density=True, histtype='step',color='k',alpha=0.8)
#----pdf分布-----
f_ax3.plot(x1, y1, color='k',alpha=0.8,lw=2)
#------2021年txx7_ano-------
plt.text(7.7,0.38,'2021',rotation='vertical',fontsize=20,color='tab:red')
#f_ax2.set_ylim(0,0.8)
#f_ax2.set_yticks(np.arange(0,1.0,0.2))
xminorLocator   = MultipleLocator(1)
f_ax3.xaxis.set_minor_locator(xminorLocator)
#yminorLocator   = MultipleLocator(0.1)
#f_ax2.yaxis.set_minor_locator(yminorLocator)
f_ax3.tick_params(labelsize=25) 
#f_ax2.legend(fontsize=15,loc='upper right')

f_ax4 = fig1.add_axes([0.1,0.18,0.45,0.3],projection=proj)
f_ax4.set_title('(c) Z500 anomaly',loc='left',fontsize=25)
f_ax4.tick_params(labelsize=25) 
f_ax4.add_patch(Polygon([[-125,40], [-105, 40], [-105, 65], [-125, 65], [-125,40]], closed=True, fill=False,color='purple',linewidth=1))
#f_ax1.add_patch(Polygon([[-180,30], [-60, 30], [-60, 85], [-180, 85], [-180,30]], closed=True, fill=False,color='k',linewidth=1))
contour_map(f_ax4,img_extent,30,30)
c = plot_filled(f_ax4,lon,lat,z500_ano,clevs1,cmaps.temp_19lev)

loca1 = fig1.add_axes([0.15,0.12,0.35,0.012])
cb    = fig1.colorbar(c,orientation='horizontal',cax=loca1,shrink=0.6,pad=0.12)
font = {
		'color'  : 'k',
        'weight' : 'normal',
        'size'   : 22,
        }
cb.set_label('colorbar',fontdict=font) #设置colorbar的标签
cb.set_label('gpm',loc='right',fontdict=font)
cb.ax.tick_params(labelsize=22)  #设置色标刻度字体大小

fig1.savefig('../pic/z500_anomaly_txx7_era5_cesm1_lastm.svg',bbox_inches='tight',dpi=600)

