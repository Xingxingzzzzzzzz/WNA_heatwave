'''
1.calculate the z500 anomaly（2021） 
2.plot the anomaly(shading) and geopotential height(contour) 
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
import os
from fnmatch import fnmatch, fnmatchcase
from matplotlib.patches import Polygon
import netCDF4 as nc
import scipy
import cmaps
from mpl_toolkits.axes_grid1 import AxesGrid
import pandas as pd
from cartopy.mpl.geoaxes import GeoAxes

plt.rc('font',family='Arial')

#===================读取z500===================
#1959-2021
#同一个变量不同时间
path = "/WORK2/zhangx/program/flow_ana/data2/ERA5/z500/"
# 读取该文件夹下的nc文件
nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".nc")]
print(nc_files)
# 读取nc文件，这里直接将多个文件按照时间维度合并
ds = xr.open_mfdataset(nc_files,combine="by_coords")
print(ds)

z500   = ds.z.sel(time=slice("1981-05-01","2010-08-31"),lat=slice(0,90),lon=slice(180,360))/9.8
z500   = z500.isel(time=z500.time.dt.month.isin([6,7])) #只选取6 7月份的数据
print(z500)
z500 = z500.convert_calendar('365_day')  #去除闰年数据，尽管这里不包含2.29，但这样时间才能对应
#日的气候态
z500_clim = z500.groupby("time.dayofyear").mean("time")
print(z500_clim)

z500_21   = ds.z.sel(time=slice("2021-06-01","2021-07-31"),lat=slice(0,90),lon=slice(180,360))/9.8

z500_21_ano = (z500_21.groupby("time.dayofyear") - z500_clim).drop("dayofyear") #求完距平后有多余的维度信息，可以drop去掉
lon  = z500_21.coords['lon']
lat  = z500_21.coords['lat']
print(z500_21_ano)


#----原来的方法----
#i=0
#j=1959
#year1  = np.arange(1959,2022,1)
#nyr    = len(year1)
#z500   = np.zeros((nyr,61,91,180))  #5-8月
#files2 = os.listdir('/WORK2/zhangx/program/flow_ana/data2/ERA5/z500/')
#files2.sort() #文件排序
#
#for file_name in files2:
#	# 读取单个文件内容
#	if fnmatch(file_name,'*.nc'):
#		print(file_name)
#		ff2 = xr.open_dataset('/WORK2/zhangx/program/flow_ana/data2/ERA5/z500/'+file_name)
#		##处理单个文件(调用方法)
#		z               = ff2.z
#		z500[i,:,:,:]   = z.isel(time=z.time.dt.month.isin([6,7])).loc[:,0:90,180:360]/9.8
#		if i ==1:
#			print(z.isel(time=z.time.dt.month.isin([6,7])).loc[:,0:90,180:360])
#		i=i+1
#
#lon = ff2['lon'].loc[180:360]
#lat = ff2['lat'].loc[0:90]
#ny  = len(lat)
#nx  = len(lon)
#
##====================z500异常（每天相对于1981-2010的异常）==================
##z500  = z500.reshape((nyr,123,ny,nx))
#z500_mean  = np.nanmean(z500[22:52,:,:,:],axis=0)  #多年平均，(123,ny,nx)
#z500_ano   = np.zeros((nyr,61,ny,nx))
#for i in range(0,nyr):
#	z500_ano[i,:,:,:] = z500[i,:,:,:] - z500_mean[:,:,:]  #每年每天的异常值
#print(z500_ano.shape)
#
#z500_21_ano = z500_ano[-1,:,:,:]
#z500_21     = z500[-1,:,:,:]
#print(z500_21_ano.shape)
#print(z500_21)
#print(np.nanmax(z500_21_ano))
#print(np.nanmin(z500_21_ano))

#========plot===============
proj     = ccrs.PlateCarree()
leftlon, rightlon, lowerlat, upperlat = (-180,-60, 0, 80)        
img_extent = [leftlon, rightlon, lowerlat, upperlat]

fig  = plt.figure(figsize=(15,15))
ax1  = fig.add_axes([0.05, 0.70, 0.3, 0.3],projection = proj)
ax2  = fig.add_axes([0.43, 0.70, 0.3, 0.3],projection = proj)
ax3  = fig.add_axes([0.81, 0.70, 0.3, 0.3],projection = proj)
ax4  = fig.add_axes([0.05, 0.45, 0.3, 0.3],projection = proj)
ax5  = fig.add_axes([0.43, 0.45, 0.3, 0.3],projection = proj)
ax6  = fig.add_axes([0.81, 0.45, 0.3, 0.3],projection = proj)
ax7  = fig.add_axes([0.05, 0.20, 0.3, 0.3],projection = proj)
ax8  = fig.add_axes([0.43, 0.20, 0.3, 0.3],projection = proj)
ax9  = fig.add_axes([0.81, 0.20, 0.3, 0.3],projection = proj)

f_ax = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]

i   = 0
n   = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
clevs1 = np.arange(-240,270,30)
clevs2 = np.arange(5320,6040,80)

for i in range(9):
	contour_map(f_ax[i],img_extent,30,20)
	f_ax[i].tick_params(labelsize=20) 

	c  = f_ax[i].contourf(lon, lat, z500_21_ano[24+i,:,:].data,zorder=0,levels=clevs1,transform=ccrs.PlateCarree(),extend='both',cmap=cmaps.temp_19lev) 
	c1 = f_ax[i].contour(lon,lat,z500_21[24+i,:,:],zorder=1,levels=clevs2,colors='k',linewidths=1,transform=ccrs.PlateCarree())
	f_ax[i].clabel(c1, c1.levels, inline=True, fontsize=18,fmt="%d",colors='k')
	#ax.set_title(f'{n[i]} ',loc='left',fontsize=12)
	if i <6:
		f_ax[i].set_title(f'{n[i]} 2021-06-{str(int(i+25)).zfill(2)}',loc='left',fontsize=20)
	else:
		f_ax[i].set_title(f'{n[i]} 2021-07-{str(int(i-5)).zfill(2)}',loc='left',fontsize=20)
	i = i+1

font = {
         'weight': 'normal',
        'color':  'k', 
        'size' :20}
location = fig.add_axes([0.38,0.20,0.4,0.01])
cb= fig.colorbar(c,orientation='horizontal',cax=location,shrink=0.6,pad=0.06)
cb.set_label('gpm',loc='right',fontdict=font)
cb.ax.tick_params(labelsize=20)

fig.savefig('../pic/z500_625-703.svg',bbox_inches='tight',dpi=600)

#projection = ccrs.PlateCarree() #投影方式
#axes_class = (GeoAxes,dict(map_projection=projection)) #建立坐标系
#fig = plt.figure(figsize=[12,12]) #建立画布，注意figsize设置大些可以防止多图重叠
#axgr = AxesGrid(fig, 111, axes_class=axes_class,
#                    nrows_ncols=(3, 3),  #5行3列
#                    axes_pad=(0.7,0.7), #水平间距，垂直间距
#                    #cbar_location='bottom', #色标位置（在右侧）
#                    #cbar_mode='single',  
#                    #cbar_pad=0.05,
#                    #cbar_size='2%',
#                    label_mode='' )
#i   = 0
#n   = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
#clevs1 = np.arange(-240,270,30)
#clevs2 = np.arange(5320,6040,80)
#
#for ax in axgr:
#	contour_map(ax,img_extent,30,20)
#	ax.tick_params(labelsize=15) 
#
#	c  = ax.contourf(lon, lat, z500_21_ano[24+i,:,:],zorder=0,transform=ccrs.PlateCarree(),extend='both',cmap=cmaps.temp_19lev) 
#	c1 = ax.contour(lon,lat,z500_21[24+i,:,:],zorder=1,levels=clevs2,colors='k',linewidths=1,transform=ccrs.PlateCarree())
#	ax.clabel(c1, c1.levels, inline=True, fontsize=12,fmt="%d",colors='k')
#	#ax.set_title(f'{n[i]} ',loc='left',fontsize=12)
#	if i <6:
#		ax.set_title(f'{n[i]} 2021-06-{str(int(i+25)).zfill(2)}',loc='left',fontsize=15)
#	else:
#		ax.set_title(f'{n[i]} 2021-07-{str(int(i-5)).zfill(2)}',loc='left',fontsize=15)
#	i = i+1
#
#font = {'style': 'italic',
#         'weight': 'normal',
#        'color':  'k', 
#        'size' :15}
#location = fig.add_axes([0.31,0.16,0.4,0.01])
#cb= fig.colorbar(c,orientation='horizontal',cax=location,shrink=0.6,pad=0.06)
#cb.set_label('gpm',loc='right',fontdict=font)
#cb.ax.tick_params(labelsize=15)
#
#fig.savefig('../pic/z500_625-703.pdf',bbox_inches='tight')
#print(okk)
#
