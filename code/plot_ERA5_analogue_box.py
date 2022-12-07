'''
对1000次取样得到的环流相似的温度和高压异常做合成，盒须图
'''
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
sys.path.append('/WORK2/zhangx/program/def')
from script_for_NAheatdome import cal_func

#from Cal import Cal_class
from plot_func import contour_map
from plot_func import xy_line
from plot_func import plot_filled
import os
from fnmatch import fnmatch, fnmatchcase
from matplotlib.patches import Polygon
import netCDF4 as nc
from scipy import signal
import cmaps
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import scipy.stats as stats

plt.rc('font',family='Arial')


#=======================
#read data
ds1      = xr.open_dataset('/WORK2/zhangx/program/flow_ana/data2/ERA5/ano_tmax/ERA5_daily_tmx_anomaly_1959-2021.nc')
tmax_ano = ds1.tmx_ano
print(tmax_ano)
lat      = tmax_ano.coords['lat']
lon      = tmax_ano.coords['lon']

#对未去趋势的温度先做去趋势(三维场，每个格点分别去趋势),给出此次事件的空间场
tmax_ano_detrend = cal_func.detrend_fit(tmax_ano)

#计算2021年的温度异常(detrend)
weights     = np.cos(np.deg2rad(tmax_ano.sel(lat=slice(40,65),lon=slice(235,255)).lat))
tmax_ano_ts = tmax_ano.sel(lat=slice(40,65),lon=slice(235,255)).weighted(weights).mean(("lat","lon")) #原时间序列，未去趋势
#detrend
tmax_ano_ts2 = cal_func.detrend_fit(tmax_ano_ts) #这里是对区域平均的序列去趋势！
tmax_ano_21  = tmax_ano_ts2.sel(time=slice('2021-06-27','2021-07-03')).mean('time')
print(f'2021 the tmax anomaly is {tmax_ano_21.data:.2f}') 

ds2          = xr.open_dataset('/WORK2/zhangx/program/flow_ana/data2/ERA5/ano_eddyz500/ERA5_daily_eddyz500_anomaly_1959-2021.nc')
eddyz500_ano = ds2.eddyz500_ano.sel(time=slice('2021-06-27','2021-07-03'))
print(eddyz500_ano)

#-----对7天的温度和高压求解平均，得到空间模态，与环流相似合成得到的作比较-----
tmax_ano_raw     = tmax_ano_detrend.sel(time=slice('2021-06-27','2021-07-03')).mean("time")
eddyz500_ano_raw = eddyz500_ano.mean("time")


#=========基于环流相似挑选得到的温度异常============
filepath = '/WORK2/zhangx/program/flow_ana/data2/ANA/'
ds1      = xr.open_dataset(filepath+'ERA5/1ERA5_ana_tmax_eddyz500_ano_pattern_1959-2020.nc')
ta       = ds1.ta  #(1000,)
eddyz500_ano_ana = ds1.eddyz500_ano #(1000,ny,nx)
tmax_ano_ana      = ds1.tmax_ano     #(1000,ny,nx)
print(ta)


ta_uchronic   = np.array(ta)
ta_m_uchronic = np.nanmean(ta_uchronic)    
print(f'ERA5 mean={ta_m_uchronic:.2f}')

ta_median = np.nanmedian(ta_uchronic)      
print(f'median={ta_median:.2f}')

r  = ta_median/tmax_ano_21*100   
print(f'{r.data:.2f}%')                   

#逐点去趋势得到的温度异常和区域平均去趋势的温度异常一样吗？（一样！
ta_uchronic2   = tmax_ano_ana.sel(lat=slice(40,65),lon=slice(235,255)).weighted(weights).mean(("lat","lon")) 
ta_median      = np.nanmedian(ta_uchronic2)      
print(f'test！！！median={ta_median:.2f}') #3.67

#区域平均的序列去趋势得到的温度异常
pro_75      = stats.scoreatpercentile(ta_uchronic, 75)
pro_25      = stats.scoreatpercentile(ta_uchronic, 25)
print(f'era5 median is {np.nanmedian(ta_uchronic):.2f} ({pro_25:.2f}-{pro_75:.2f})')

#2021 the tmax anomaly is 6.71
#era5 median is 3.67 (3.36-3.97)
#54.64%

#相似的空间场
tmax_ano_ana  = tmax_ano_ana.mean("n")
eddyz_ano_ana = eddyz500_ano_ana.mean("n")

#============随意挑选的温度异常，控制实验==============
ds2         = xr.open_dataset(filepath+'ERA5/1ERA5_ana_tmax_ano_randomly_1959-2020.nc')
ta_randomly = ds2.ta_r
ta_randomly = np.array(ta_randomly)

#======plot=======
data = [ta_uchronic,ta_randomly]

fig1  = plt.figure(figsize=(15,15)) 

f_ax4 = fig1.add_axes([0.12,0.8,0.75,0.25])

#color = dict(boxes='DarkGreen', whiskers='DarkGreen', medians='DarkGreen', caps='DarkGreen')
# 箱型图着色
# boxes → 箱线
# whiskers → 分位数与error bar横线之间竖线的颜色
# medians → 中位数线颜色
# caps → error bar横线颜色
color = plt.cm.tab20c(np.linspace(0,1,20))

bp = f_ax4.boxplot(data,labels=('Analogs', 'Control'),
	           showmeans = False,
	           widths    = 0.3,
	           showbox   = True,
	           patch_artist = False, #填充颜色
               boxprops = {'color':color[0],'linewidth':2}, # 设置箱体属性，如边框色和填充色
              # 设置异常点属性，如点的形状、填充色和点的大小
              flierprops = {'marker':'o', 'markersize':3,'markeredgecolor':color[0]}, 
              # 设置均值点的属性，如点的形状、填充色和点的大小
              #meanprops = {'marker':'D','markerfacecolor':'indianred', 'markersize':4}, 
              # 设置中位数线的属性，如线的类型和颜色
              medianprops = {'color':color[0],'linewidth':2},
              capprops    = {'color':color[0],'linewidth':2},		#capprops：设置箱线图顶端和末端线条的属性，如颜色、粗细等
              whiskerprops= {'color':color[0],'linestyle':'-','linewidth':2})		#设置须的属性，如颜色、粗细、线的类型等

#df.plot.box(vert=True, 
#             positions=[1, 4],
#             ax = ax,
#             grid = True,
#             color = color,
#             labels=('Analogues', 'Control'))
f_ax4.set_title('(a)',loc='left',fontsize=22)
f_ax4.set_title(f'{r.data:.2f}%',loc='right',fontsize=22)
f_ax4.axhline(0,linestyle='-',linewidth=1,color='k')
f_ax4.axhline(tmax_ano_21,linestyle='--',color='r',linewidth=1.5)
f_ax4.set_ylabel('Tmax anomaly ($^\circ$C)',fontsize=22)
f_ax4.set_ylim(-6,9)
f_ax4.set_yticks(np.arange(-6,12,3))
yminorLocator   = MultipleLocator(1)
f_ax4.yaxis.set_minor_locator(yminorLocator)
f_ax4.tick_params(labelsize=22) 


proj  = ccrs.PlateCarree()
leftlon, rightlon, lowerlat, upperlat = (-160,-60, 20, 80)        
img_extent = [leftlon, rightlon, lowerlat, upperlat]

clevs2 = np.arange(-200,240,40)
f_ax1 = fig1.add_axes([0.05,0.41,0.4,0.4],projection=proj)
f_ax1.set_title('(b) Eddy z500',loc='left',fontsize=22)
f_ax1.add_patch(Polygon([[-125,40], [-105, 40], [-105, 65], [-125, 65], [-125,40]], closed=True, fill=False,color='purple',linewidth=1))
contour_map(f_ax1,img_extent,20,20)
f_ax1.tick_params(labelsize=22) 
c1 = plot_filled(f_ax1,eddyz500_ano.coords['lon'],eddyz500_ano.coords['lat'],eddyz500_ano_raw,clevs2,cmaps.temp_19lev)

f_ax2 = fig1.add_axes([0.05,0.1,0.4,0.4],projection=proj)
f_ax2.set_title('(d) Analogue eddy z500',loc='left',fontsize=22)
f_ax2.add_patch(Polygon([[-125,40], [-105, 40], [-105, 65], [-125, 65], [-125,40]], closed=True, fill=False,color='purple',linewidth=1))
contour_map(f_ax2,img_extent,20,20)
f_ax2.tick_params(labelsize=22) 
c2 = plot_filled(f_ax2,eddyz500_ano.coords['lon'],eddyz500_ano.coords['lat'],eddyz_ano_ana,clevs2,cmaps.temp_19lev)

font = {
         'weight': 'normal',
        'color':  'k', 
        'size' :22
        }
loca2  = fig1.add_axes([0.075,0.11,0.35,0.012])
cb2    = fig1.colorbar(c2,orientation='horizontal',cax=loca2,shrink=0.6,pad=0.12)
cb2.set_label('gpm',loc='right',fontdict=font)
cb2.ax.tick_params(labelsize=22)


clevs1 = np.arange(-11,13,2)
f_ax3 = fig1.add_axes([0.55,0.41,0.4,0.4],projection=proj)
f_ax3.set_title('(c) Tmax',loc='left',fontsize=22)
f_ax3.add_patch(Polygon([[-125,40], [-105, 40], [-105, 65], [-125, 65], [-125,40]], closed=True, fill=False,color='purple',linewidth=1))
contour_map(f_ax3,img_extent,20,20)
f_ax3.tick_params(labelsize=22) 
c = plot_filled(f_ax3,tmax_ano.coords['lon'],tmax_ano.coords['lat'],tmax_ano_raw,clevs1,cmaps.CBR_coldhot)

f_ax4 = fig1.add_axes([0.55,0.1,0.4,0.4],projection=proj)
f_ax4.set_title('(e) Analogue tmax',loc='left',fontsize=22)
f_ax4.add_patch(Polygon([[-125,40], [-105, 40], [-105, 65], [-125, 65], [-125,40]], closed=True, fill=False,color='purple',linewidth=1))
contour_map(f_ax4,img_extent,20,20)
f_ax4.tick_params(labelsize=22) 
c = plot_filled(f_ax4,tmax_ano.coords['lon'],tmax_ano.coords['lat'],tmax_ano_ana,clevs1,cmaps.CBR_coldhot)

loca1 = fig1.add_axes([0.575,0.11,0.35,0.012])
cb    = fig1.colorbar(c,ticks=np.arange(-10,12,2),orientation='horizontal',cax=loca1,shrink=0.6,pad=0.12)
cb.set_label('$^\circ$C',loc='right',fontdict=font)
cb.ax.tick_params(labelsize=22)

fig1.savefig('../pic/ERA5_Analogues_z500_tmax_box_cubic.svg',bbox_inches='tight',dpi=600)
