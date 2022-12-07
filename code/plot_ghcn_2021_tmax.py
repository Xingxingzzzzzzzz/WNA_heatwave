'''
1.pick up the station in (20-80N, 160W-60W)
2.2021 data pick up the tmax data in NA_NW, write as ghcn_sta_tmax_2021.csv
3.plot the station map

'''

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import sys
sys.path.append('/WORK2/zhangx/program/def')
from plot_func import contour_map
from plot_func import truncate_colormap
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.patches import Polygon
import netCDF4 as nc
import cmaps
import matplotlib.colors as mcolors

plt.rc('font',family='Arial')

#===============提取北美西北部范围内的站点================
#data = pd.read_csv('../data2/ghcn_sta/ghcnd-stations.txt', sep='\s+',header=None,names=['sta','lat','lon','elevation','state','name','gsn','hcn/crn'])
#sta  = data.sta
#lon  = data.lon
#lat  = data.lat
#
#lat  = np.array(lat)
#lon  = np.array(lon)
#
#j       = 0
#array   = np.zeros((76421))
#new_sta = list(array)
#lon2    = np.zeros((76421))
#lat2    = np.zeros((76421))
#for i in range(0,len(lat)):
#	if ((lat[i]>=20 and lat[i]<=80) and (lon[i]<=-60 and lon[i]>=-160)):
#		#print(f'station is {sta[i]}')
#		#print(f'lat={lat[i]},lon={lon[i]}')
#		new_sta[j] = sta[i]
#		lon2[j]    = lon[i]
#		lat2[j]    = lat[i]
#		j =j+1    #先计算得到的j有76421个
#
#print(j)
#
##=========存储数据=========
#dd2       = {'station':new_sta,'lat':lat2,'lon':lon2}
#d2frame   = pd.DataFrame(dd2,index=np.arange(0,76421,1))
#print(d2frame)
#
##将DataFrame存储为csv,index表示是否显示行名，default=True
##d2frame.to_csv(r"./data2/ghcn_sta/ghcn_sta_NA_NW.csv",sep=',')
#
#
##=========2021年的最高温，把北美西部的站点数据均挑出来============
#df      = pd.read_csv('/WORK2/zhangx/data/GHCN/2021.csv', sep=',',header=None,names=['sta','time','element','value','a','b','c','d'])
#print(df)
#station = df.sta[(df.element=='TMAX')] #Tmax站点数据，全球
#
##yr   = np.arange(1950,2022,1)
#time_all = list(pd.date_range('2021-06-01','2021-08-31',freq='D'))
#ntime    = len(time_all)
#df       = df.set_index('sta')
#
#temp       = df.value[(df.element=='TMAX')]  #所有站点所有时间（6.1-8.31）
#time       = df.time[(df.element=='TMAX')]
#print(temp)
#print(time)
#
#j=0
#for staa in new_sta:
#	print(staa)
#	if staa in list(station):
#		tmax     = temp.loc[staa]   #北美站点
#		time2    = time.loc[staa]
#		n    = new_sta.index(staa)
#		lat3 = lat2[n]
#		lon3 = lon2[n]
#		print(type(tmax))
#		if type(tmax)!=np.int64:  #加这一句因为如果只选出一个数字会出错！
#			print(len(tmax))
#			if len(tmax)==92:   
#				print(f'selected station is {j}')
#				if j==0:
#					dd        = {'station':staa,'lat':lat3,'lon':lon3,'time':time_all,'tmax':tmax}
#					data2     = pd.DataFrame(dd) #index为行标签
#					print(data2)
#					j = j+1
#				if j >=1:
#					dd       = {'station':staa,'lat':lat3,'lon':lon3,'time':time_all,'tmax':tmax}
#					data2    = data2.append(pd.DataFrame(dd))
#					j = j+1
#					print(data2)
#			else:
#				print('the data is not statisfied')
#		else:
#			print('one data!!!!!!!!!!!')
#		del(tmax)
#		del(time2)
#	else:
#		print(f'the {staa} is not in station')
#
#data2.to_csv(r"/WORK2/zhangx/program/flow_ana/data2/ghcn_sta/ghcn_sta_tmax_2021.csv",sep=',',index=None)

#=======读取数据=======
df    = pd.read_csv("/WORK2/zhangx/program/flow_ana/data2/ghcn_sta/ghcn_sta_tmax_2021.csv",sep=',',skiprows=0)
df.columns = ['sta','lat','lon','date','tmax']
print(df)

nsta =  int(len(df.tmax)/92)  #站点数4855个站点
nday = 92
allday = len(df.tmax)

#变为数组形式
temp = np.zeros((nsta,92))   
lon  = np.zeros((nsta,92))
lat  = np.zeros((nsta,92))
for i in range(0,92):
	temp[:,i]  = df.tmax[i:allday:92]/10
	lon[:,i]   = df.lon[i:allday:92]
	lat[:,i]   = df.lat[i:allday:92]

print(np.nanmax(temp))
print(np.nanmin(temp))

##站点很多重叠的！所以保留站点间隔大于2度的站点
tmax     = temp
lon_new  = lon
lat_new  = lat
n        = 1 
for i in range(0,nsta):
	for j in range(i+1,nsta):
		if ((lon[i,0]-2.0<lon[j,0]) & (lon[j,0]<lon[i,0]+2.0) & (lat[i,0]-2.0<lat[j,0]) & (lat[j,0]<lat[i,0]+2.0)).all():
			lon_new[j,:] = lon[i,:]
			lat_new[j,:] = lat[i,:] 
			tmax[j,:]    = temp[i,:]
			
print(tmax)
print(lon_new[:,0])
#


font = {
         'weight': 'normal',
        'color':  'k', 
        'size':15
        }

######站点图
matplotlib.use('Agg')
figsize = (10, 8)
cols = 5   #列数
rows = 6
proj     = ccrs.PlateCarree()
leftlon, rightlon, lowerlat, upperlat = (-180,-60, 0, 80)        
img_extent = [leftlon, rightlon, lowerlat, upperlat]

projection = ccrs.PlateCarree() #投影方式
axes_class = (GeoAxes,dict(map_projection=projection)) #建立坐标系
fig = plt.figure(figsize=[12,12]) #建立画布，注意figsize设置大些可以防止多图重叠
axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(3, 3),  #5行3列
                    axes_pad=(0.7,0.7), #水平间距，垂直间距
                    #cbar_location='bottom', #色标位置（在右侧）
                    #cbar_mode='single',  
                    #cbar_pad=0.05,
                    #cbar_size='2%',
                    label_mode='' )

new_cmap = truncate_colormap('RdBu_r', minval=0.5, maxval=1.0,n=50)
uneven_levels = [0, 20, 25, 30, 35, 40, 45, 50,55]
cmap_rb = plt.get_cmap(new_cmap)
colors  = cmap_rb(np.linspace(0, 1, len(uneven_levels) - 1))
cmap, norm = mcolors.from_levels_and_colors(uneven_levels, colors)


i   = 0
n   = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
clevs1 = np.arange(0,50,1)
for ax in axgr:
	contour_map(ax,img_extent,30,20)
	ax.tick_params(labelsize=15) 
	c = ax.scatter(lon_new[:,24+i],lat_new[:,24+i],c=tmax[:,i+24],marker=".", s=30, cmap=cmap, norm=norm,transform=ccrs.PlateCarree())
	#p = ax.contourf(lon[:,6716+i],lat[:,6716+i],ta_2021[:,i],bounds,transform=projection,norm=my_norm,cmap=my_map)
	#ax.add_patch(Polygon([[-140,40], [-100, 40], [-100, 70], [-140, 70], [-140,40]], closed=True, fill=False,color='tab:blue',linewidth=1))
	#ax.set_title(f'({n[i]}) tmax anomaly',loc='left',fontsize=15)
	if i <6:
		ax.set_title(f'{n[i]} 2021-06-{str(int(i+25)).zfill(2)}',loc='left',fontsize=15)
	else:
		ax.set_title(f'{n[i]} 2021-07-{str(int(i-5)).zfill(2)}',loc='left',fontsize=15)
	i = i+1


location = fig.add_axes([0.32,0.15,0.4,0.01])
cb= fig.colorbar(c,orientation='horizontal',cax=location,shrink=0.6,pad=0.06)
cb.set_label('$^\circ$C',loc='right',fontdict=font)
cb.ax.tick_params(labelsize=15)
fig.savefig('../pic/station_map_tmax_2021.svg',bbox_inches='tight',dpi=600)
