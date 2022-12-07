import numpy as np
import xarray as xr
import os
import netCDF4 as nc
from scipy.stats import genextreme as gev, kstest
from scipy import stats
import openturns as ot
import random
from   datetime  import  *
import  time
import pandas as pd
from sklearn.utils import resample
from scipy.optimize import leastsq
import cal_function_from_jiang
from csaps import csaps



class cal_func(object):
	"""docstring for cal_func"""
	def __init__(self, data):
		super(cal_func, self).__init__()
		self.data = data


#================================
#计算anomaly
#================================
	def cal_anomaly(var,time1,time2,mask): #time1-time2为计算气候态的时间段
		#if varname=='hgt':
		#	print('ERA5 Z500!!!!!')
		#	var = ds.varname/9.8  #ERA5 z500
		#elif ((varname=='mx2t') or (varname=='TREFHTMX')):
		#	print('tmax!!!!!')
		#	var = ds.varname0-273.15 #ERA5 tmax or CESM TREFHTMX
		#else:
		#	var = ds.varname
		#print(var)
		#select data,时间范围和经纬度范围
		var    = var.sel(lat=slice(20,80),lon=slice(200,300))
		#计算气候态
		sel_ts   = dict(time=slice(time1,time2))   #气候态的时间段,比如1981-2010
		var_clim = var.sel(**sel_ts).groupby("time.dayofyear").mean("time")
		print(var_clim)
		#计算anomaly
		var_ano  = (var.groupby("time.dayofyear")-var_clim).drop("dayofyear") #求完距平后有多余的维度信息，可以drop去掉
		#print(var_ano)
		#print(var_ano.data)

		#mask
		if mask=='True':
			print('mask!!!')
			ff3       = nc.Dataset('/WORK2/zhangx/data/ERSST/v5/ersst.v5.1900.2019.1x1.nc')       
			sst_mask  = ff3['sst'][0,0,110:171,200:301]
			#print(sst_mask)
			var_ano   = xr.where(sst_mask.mask==True,var_ano,np.nan)

		return var_ano

	def txxn_eddyz(var,cir,nn,pattern): #计算TXX7以及对应的eddyz500

		#只针对夏季6,7,8月
		var = var.isel(time=var.time.dt.month.isin([6,7,8])).sel(lat=slice(20,80),lon=slice(200,300))
		cir = cir.isel(time=cir.time.dt.month.isin([6,7,8])).sel(lat=slice(20,80),lon=slice(200,300))
		print(var)
		#-----区域平均(如北美40:65N,125W:105W)-----
		# 定义纬度权重
		weights = np.cos(np.deg2rad(var.sel(lat=slice(40,65),lon=slice(235,255)).lat))
		#计算加权平均
		var_area = var.sel(lat=slice(40,65),lon=slice(235,255)).weighted(weights).mean(("lat","lon"))
		cir_area = cir.sel(lat=slice(40,65),lon=slice(235,255)).weighted(weights).mean(("lat","lon"))
		print(var_area)
		
		#-----滑动平均--------
		#print(var_area.groupby('time.year').groups)
		#var_rm  = var_area.reshape((nyr,nday)).rolling(time=n, center=True).mean().dropna("time")
		#cir_rm  = cir_area.reshape((nyr,nday)).rolling(time=n, center=True).mean().dropna("time")
		#print(var_rm)
		#txx7    = var_rm.groupby('time.year').max()
		#print(txx7)
		#cir_xx7 = np.zeros((txx7.data.shape))

		a        = np.repeat(1/nn,nn)
		nyr      = int(var_area.data.shape[0]/92)
		nday     = 92
		nday2    = nday-(nn-1)
		var_rm   = np.zeros((nyr,nday2))
		cir_rm   = np.zeros((nyr,nday2))
		var_area = np.array(var_area).reshape(nyr,nday)
		cir_area = np.array(cir_area).reshape(nyr,nday)
		for y in range(0,nyr):
			var_rm[y,:] = np.convolve(var_area[y,:],a,mode='valid')   #7天滑动平均所以前后去掉了3天！！！！！
			cir_rm[y,:] = np.convolve(cir_area[y,:],a,mode='valid')
		#找出每年JJA中tmax(基本场) 7天滑动平均最大的值作为该年的var_Xx7，并把对应的7天滑动平均的eddyz500也取出来
		if pattern=='True':
			var_xx7_3d = np.full((nyr,var.data.shape[1],var.data.shape[2]),np.nan)  #var是20:80N,200:300
			cir_xx7_3d = np.full((nyr,var.data.shape[1],var.data.shape[2]),np.nan)
			var_xx7    = np.zeros((nyr))
			cir_xx7    = np.zeros((nyr))
			for y in range(0,nyr):
				var_xx7[y] = np.nanmax(var_rm[y,:])   
				n          = list(var_rm[y,:]).index(var_xx7[y])
				cir_xx7[y] = cir_rm[y,n]
				var_xx7_3d[y,:,:]= np.nanmean(var.data.reshape(nyr,nday,var.data.shape[1],var.data.shape[2])[y,n:n+7,:,:],axis=0)    #对应的涡动位势高度和高温异常（空间型）这里确实是n:n+7!!!
				cir_xx7_3d[y,:,:]= np.nanmean(cir.data.reshape(nyr,nday,var.data.shape[1],var.data.shape[2])[y,n:n+7,:,:],axis=0)     #涡动位势高度异常！！
			print(var_xx7)
			print(cir_xx7)
			return var_xx7,cir_xx7,var_xx7_3d,cir_xx7_3d
		else:
			var_xx7    = np.zeros((nyr))
			cir_xx7    = np.zeros((nyr))
			for y in range(0,nyr):
				var_xx7[y] = np.nanmax(var_rm[y,:])   
				n          = list(var_rm[y,:]).index(var_xx7[y])
				cir_xx7[y] = cir_rm[y,n]
			print(var_xx7)
			print(cir_xx7)
			return var_xx7,cir_xx7


	def txxn(var,nn,day=None): #计算TXX7
	
		#只针对夏季6,7,8月
		var = var.isel(time=var.time.dt.month.isin([6,7,8])).sel(lat=slice(20,80),lon=slice(200,300))
		print(var)
		#-----区域平均(如北美40:65N,125W:105W)-----
		# 定义纬度权重
		weights = np.cos(np.deg2rad(var.sel(lat=slice(40,65),lon=slice(235,255)).lat))
		#计算加权平均
		var_area = var.sel(lat=slice(40,65),lon=slice(235,255)).weighted(weights).mean(("lat","lon"))
		print(var_area)
		
		#-----滑动平均--------
		a        = np.repeat(1/nn,nn)
		nyr      = int(var_area.data.shape[0]/92)
		nday     = 92
		nday2    = nday-(nn-1)
		var_rm   = np.zeros((nyr,nday2))
		var_area = np.array(var_area).reshape(nyr,nday)
		for y in range(0,nyr):
			var_rm[y,:] = np.convolve(var_area[y,:],a,mode='valid')   #7天滑动平均所以前后去掉了3天！！！！！
		#找出每年JJA中tmax(基本场) 7天滑动平均最大的值作为该年的var_Xx7，并把对应的7天滑动平均的eddyz500也取出来
		var_xx7    = np.zeros((nyr))
		n          = np.zeros((nyr))
		for y in range(0,nyr):
			var_xx7[y] = np.nanmax(var_rm[y,:])   
			#同时给出txx7最大时对应的索引(夏季)
			if day is not None:
				n[y] = list(var_rm[y,:]).index(var_xx7[y])  #如果得到的索引是0，那就是0-6天的温度最高,这里是每年的索引！
		print(var_xx7)
		if day is not None:
			print(n[-1])
			return var_xx7,n
		else:
			return var_xx7

#----------------------------------
#GEV
#----------------------------------
	def gevv(self,d):
		data_s = sorted(self.data)
		shape, loc, scale = gev.fit(data_s)
		print(f'shape is {shape}')
		print(f'loc is {loc}')
		print(f'scale is {scale}')
		a,p = kstest(data_s, 'genextreme', gev.fit(data_s))
		if p > 0.05:
			print('The data follows GEV distribution!' )
	
		l = loc + scale / shape
	
		xx = np.linspace(data_s[0]-d,data_s[-1]+d, num=1000)
		yy = gev.pdf(xx, shape, loc, scale)
	
		return xx,yy,p


	def gevv2(self):
		data_s = sorted(self.data)
		sample = ot.Sample([[p] for p in data_s])
		rr= ot.GeneralizedExtremeValueFactory().buildAsGeneralizedExtremeValue(sample)
		print(rr)

#================================
#model MME+std/MME-std or 5th-95th
#================================
	def model_sd(self,style):
		nm  = self.data.shape[0]
		nt  = self.data.shape[1]
		MME = np.nanmean(self.data,axis=0)
		if style=='MME+std':
			data_sd = np.zeros((nt))
			for i in range(nt):
				data_sd[i] = np.nanstd(self.data[:,i])
			data1 = MME - data_sd
			data2 = MME + data_sd
		else:
			data1   = np.zeros((nt))
			data2   = np.zeros((nt))
			for i in range(nt):
				data1[i]  = stats.scoreatpercentile(self.data[:,i], 5)   #计算5%分位数
				data2[i]  = stats.scoreatpercentile(self.data[:,i], 95)  #95分位数 
		return MME,data1,data2


#=====================================================================
#*************环流相似，分别在nday天nana个类中随机取样*************
#=====================================================================
	def ana_random(self,ana_day,nday,var_circulation,nn,tmax_nodetrend=None):         #ana_day挑选的环流相似的日期，ana_day[nday,nana],nday即极端事件的天数,nn-取样次数
		time 	   = self.data.coords['time']
		lat  	   = self.data.coords['lat']
		lon  	   = self.data.coords['lon']
		ny         = len(lat)
		nx         = len(lon)
		#datetimeindex = self.data.indexes['time'].to_datetimeindex()
		#self.data.coords['time'] = datetimeindex
		time2      = pd.to_datetime(time.data)
		#print(time2)
		#print(ok)

		var_ana    = np.full((nday,ny,nx),np.nan)    
		cir_ana    = np.full((nday,ny,nx),np.nan)
		ttt_ana    = np.full((nday,ny,nx),np.nan)    

		sample1    = np.zeros((nday,1),dtype=int)
		var_ana2   = np.full((nn,ny,nx),np.nan)   #nn次取样
		cir_ana2   = np.full((nn,ny,nx),np.nan)
		ttt_ana2   = np.full((nn,ny,nx),np.nan)   #不去趋势的！

		
		m=1
		for j in range(0,nn):
			print(f'{j}th sampling')
#
###--------先对日期做随机抽样（每nana(eg.20)个同类的日期挑一天,挑出nday(eg.7)个样本）------
			for i in range(0,nday):
				day2 = list(ana_day[i,:])
				#print(day2)
				sample1[i,:] = random.sample(day2,m)    #在day2中随机取1天，循环挑出nday天
			#print(f'sample date is {sample1}')
			sample2 =[datetime.strptime(str(int(x)),'%Y%m%d') for x in sample1[:,0]]  #numpy转为时间格式
			sample2 = pd.DatetimeIndex(sample2, dtype='datetime64[ns]', freq=None)    #转为datetime64        
			#print(sample2)  

##-------找出nday个日期对应的变量--------
			t = 0
			for dt in sample2:    
				#print(dt)
				if dt in time2:   #time or time2为var变量中的每一天
					#print(dt)
					time_list = list(time2)
					n  = time_list.index(dt)    #获取对应日期的下标，注：a.index()列表的用法（注：list只能找到第一个匹配的，不过这里时间都是唯一的
					#print(n)
					var_ana[t,:,:] = self.data[n,:,:]
					cir_ana[t,:,:] = var_circulation[n,:,:] #异常场(如eddy z500 ano)
					if tmax_nodetrend is not None:
						ttt_ana[t,:,:]   =  tmax_nodetrend[n,:,:]
					#print(self.data[n])
					t = t+1				
				else:
					print('error!')
	
			var_ana2[j,:,:]   = np.nanmean(var_ana,axis=0)     ##nday天平均
			cir_ana2[j,:,:]   = np.nanmean(cir_ana,axis=0)
			if tmax_nodetrend is not None:
				ttt_ana2[j,:,:]   = np.nanmean(ttt_ana,axis=0)

		if tmax_nodetrend is not None:
			return var_ana2,cir_ana2,ttt_ana2
		else:
			return var_ana2,cir_ana2

#=================================
#随机取样
#=================================
	def random(self,start_year,end_year,nday,nn,day1,day2,day3,day4):  #nday事件的天数
		time 	 = self.data.coords['time']
		lat  	 = self.data.coords['lat']
		lon  	 = self.data.coords['lon']
		ny       = len(lat)
		nx       = len(lon)

		##-------先做7天的滑动平均，这样抽到1天就代表了7天的平均(or 3天)--------
		nyr = end_year-start_year+1
		var_windows = np.full((nyr,60+nday,ny,nx),np.nan)
		##可以用loc
		for i in range(start_year,end_year+1):
			#var_windows[j:j+60+nday,:,:] = self.data.loc[str(i)+'-06-01':str(i)+'-08-30']  #把每年的事件前后30天的数据先选出来!视具体而定！！
			#var_windows[j:j+60+nday,:,:] = self.data.loc[str(i)+'-05-28':str(i)+'-08-02']
			#var_windows[j:j+60+nday,:,:] = self.data.loc[str(i)+'-12-20':str(i+1)+'-02-19']
			if int(day1[0:2])>int(day2[0:2]):   #月份比较，跨年了，所以要i+1
				print('such as DJF')
				var_windows[i-start_year,:,:,:] = self.data.loc[str(i)+'-'+day1:str(i+1)+'-'+day2]
			else:
				var_windows[i-start_year,:,:,:] = self.data.loc[str(i)+'-'+day1:str(i)+'-'+day2]
		
		#var_windows = var_windows.reshape((nyr,60+nday,ny,nx))   #因为是每年分别做滑动平均，所以reshape
		a           = np.repeat(1/nday, nday)
		var_w       = np.full((nyr,60+1,ny,nx),np.nan)
		#
		##卷积运算(滑动平均)
		for yr in range(0,nyr):
			for i in range(0,ny):
				for j in range(0,nx):
					b = np.convolve(var_windows[yr,:,i,j], a, mode='valid')  
					var_w[yr,:,i,j] = b
		var_w = var_w.reshape((nyr*(60+1),ny,nx))
		#
		##----------把每年对应的事件前后30天的时间段选出来------------
		#可以转化为pandas的datetime64即时间戳用以下程序!!
		j=0
		time3 = np.zeros((nyr*(60+1)))
		for i in range(start_year,end_year+1):
			#time3[j:j+61] = time.loc[str(i)+'-06-16':str(i)+'-08-15']        #滑动平均后砍掉了前后几天
			#time3[j:j+61]  = time.loc[str(i)+'-05-31':str(i)+'-07-30']
			#time3[j:j+61]  = time.loc[str(i)+'-12-21':str(i+1)+'-02-19']
			if int(day3[0:2])>int(day4[0:2]):   #月份比较，跨年了，所以要i+1
				time3[j:j+61]  = time.loc[str(i)+'-'+day3:str(i+1)+'-'+day4]
			else:
				time3[j:j+61]  = time.loc[str(i)+'-'+day3:str(i)+'-'+day4]
			j = j+61
		time3 = pd.to_datetime(time3)  

		##-----------在前后30天中抽一天------------
		t=0
		var_random   = np.full((nn,ny,nx),np.nan)
		for j in range(0,nn):
			#print(f'{j}th sampling')
			r_sample  = random.sample(list(time3),1)     #滑动平均了所以在前后30天抽一天即可
			r_sample2 = pd.to_datetime(r_sample)
			#print(f'random sample date is {r_sample2}')
			#print(r_sample2)
			n  = list(time3).index(r_sample2)            #看这个时间在这61天中的索引
			#print(time3[n]) 
			var_random[j,:,:] = var_w[n,:,:]
			
		return var_random

#==================================================================================================
#*************环流相似，分别在nday天nana个类中随机取样*************
#不同于ana_random,输入场为区域平均的（如WNA区域）去趋势后的温度序列，环流空间场，[未去趋势的温度序列]
#温度异常的空间场，如果要给即tmax_ano is not None，逐点去趋势
#==================================================================================================
	def ana_random_detrend(self,ana_day,nday,var_circulation,nn,tmax_nodetrend=None,tmax_ano=None):         #ana_day挑选的环流相似的日期，ana_day[nday,nana],nday即极端事件的天数,nn-取样次数
		#self-去趋势的温度的时间序列，tmax_nodetrend-温度的时间序列(未去趋势)
		#tmax_ano-未去趋势的温度(time,lat,lon)
		time 	   = self.data.coords['time']
		time2      = pd.to_datetime(time.data)
		print(time2)

		lat  	   = var_circulation.coords['lat']
		lon  	   = var_circulation.coords['lon']
		ny         = len(lat)
		nx         = len(lon)

		var_ana    = np.full((nday),np.nan)    
		ttt_ana    = np.full((nday),np.nan)  
		cir_ana    = np.full((nday,ny,nx),np.nan)  

		sample1    = np.zeros((nday,1),dtype=int)
		var_ana2   = np.full((nn),np.nan)   #nn次取样
		ttt_ana2   = np.full((nn),np.nan)   #不去趋势的！
		cir_ana2   = np.full((nn,ny,nx),np.nan)


		if tmax_ano is not None:
			#类似温度的空间场
			tmax_ana  = np.full((nday,ny,nx),np.nan)
			tmax_ana2 = np.full((nn,ny,nx),np.nan)
			#对未去趋势的温度先做去趋势(三维场，每个格点分别去趋势)
			tmax_ano_detrend = cal_func.detrend_fit(tmax_ano)
			print(tmax_ano_detrend)

		m=1
		for j in range(0,nn):
			print(f'{j}th sampling')
#
###--------先对日期做随机抽样（每nana(eg.20)个同类的日期挑一天,挑出nday(eg.7)个样本）------
			for i in range(0,nday):
				day2 = list(ana_day[i,:])
				#print(day2)
				sample1[i,:] = random.sample(day2,m)    #在day2中随机取1天，循环挑出nday天
			print(f'sample date is {sample1}')
			sample2 =[datetime.strptime(str(int(x)),'%Y%m%d') for x in sample1[:,0]]  #numpy转为时间格式
			sample2 = pd.DatetimeIndex(sample2, dtype='datetime64[ns]', freq=None)    #转为datetime64        
			#print(sample2)  

##-------找出nday个日期对应的变量--------
			t = 0
			for dt in sample2:    
				#print(dt)
				if dt in time2:   #time or time2为var变量中的每一天
					#print(dt)
					time_list = list(time2)
					n  = time_list.index(dt)    #获取对应日期的下标，注：a.index()列表的用法（注：list只能找到第一个匹配的，不过这里时间都是唯一的
					#print(n)
					var_ana[t]     = self.data[n]           #只有时间维
					cir_ana[t,:,:] = var_circulation[n,:,:] #异常场(如eddy z500 ano，空间场！)
					if tmax_ano is not None:
						tmax_ana[t,:,:] = tmax_ano_detrend[n,:,:]
					if tmax_nodetrend is not None:               #如果需要输出未去趋势的温度异常！
						ttt_ana[t]   =  tmax_nodetrend[n]
					print(self.data[n])
					t = t+1				
				else:
					print('error!')
	
			var_ana2[j]       = np.nanmean(var_ana,axis=0)     ##nday天平均
			cir_ana2[j,:,:]   = np.nanmean(cir_ana,axis=0)
			if tmax_nodetrend is not None:
				ttt_ana2[j]   = np.nanmean(ttt_ana,axis=0)
			if tmax_ano is not None:
				tmax_ana2[j,:,:] = np.nanmean(tmax_ana,axis=0)

		if tmax_nodetrend is not None:
			return var_ana2,cir_ana2,ttt_ana2
		if tmax_ano is not None:
			return var_ana2,cir_ana2,tmax_ana2
		else:
			return var_ana2,cir_ana2

	#-----------------------------------------------
	#不依赖于环流做随机取样，只给nn次取样的结果，无空间型
	def random_nn(self,start_year,end_year,nday,nn,day1,day2,day3,day4):  #nday事件的天数
		time 	 = self.data.coords['time']

		##-------先做7天的滑动平均，这样抽到1天就代表了7天的平均(or 3天)--------
		nyr = end_year-start_year+1
		var_windows = np.full((nyr,60+nday),np.nan)
		##可以用loc
		for i in range(start_year,end_year+1):
			if int(day1[0:2])>int(day2[0:2]):   #月份比较，跨年了，所以要i+1
				print('such as DJF')
				var_windows[i-start_year,:] = self.data.loc[str(i)+'-'+day1:str(i+1)+'-'+day2]
			else:
				var_windows[i-start_year,:] = self.data.loc[str(i)+'-'+day1:str(i)+'-'+day2]
		
		#var_windows = var_windows.reshape((nyr,60+nday,ny,nx))   #因为是每年分别做滑动平均，所以reshape
		a           = np.repeat(1/nday, nday)
		var_w       = np.full((nyr,60+1),np.nan)
		#
		##卷积运算(滑动平均)
		for yr in range(0,nyr):
			b = np.convolve(var_windows[yr,:], a, mode='valid')  
			var_w[yr,:] = b
		var_w = var_w.reshape((nyr*(60+1)))
		#
		##----------把每年对应的事件前后30天的时间段选出来------------
		#可以转化为pandas的datetime64即时间戳用以下程序!!
		j=0
		time3 = np.zeros((nyr*(60+1)))
		for i in range(start_year,end_year+1):
			if int(day3[0:2])>int(day4[0:2]):   #月份比较，跨年了，所以要i+1
				time3[j:j+61]  = time.loc[str(i)+'-'+day3:str(i+1)+'-'+day4]
			else:
				time3[j:j+61]  = time.loc[str(i)+'-'+day3:str(i)+'-'+day4]
			j = j+61
		time3 = pd.to_datetime(time3)  

		##-----------在前后30天中抽一天------------
		t=0
		var_random   = np.full((nn),np.nan)
		for j in range(0,nn):
			#print(f'{j}th sampling')
			r_sample  = random.sample(list(time3),1)     #滑动平均了所以在前后30天抽一天即可
			r_sample2 = pd.to_datetime(r_sample)
			#print(f'random sample date is {r_sample2}')
			#print(r_sample2)
			n  = list(time3).index(r_sample2)            #看这个时间在这61天中的索引
			#print(time3[n]) 
			var_random[j] = var_w[n]
			
		return var_random
    #=======================================================================================
    #同ana_random_detrend，但针对的是pi-control的数据，时间格式无法转为datetime64,因此用PeriodIndex
    #=======================================================================================
	def ana_random_detrend2(self,ana_day,nday,var_circulation,nn,tmax_nodetrend=None):         #ana_day挑选的环流相似的日期，ana_day[nday,nana],nday即极端事件的天数,nn-取样次数
		time 	   = self.data.coords['time']
		print(time)
		time_list = list(time)

		lat  	   = var_circulation.coords['lat']
		lon  	   = var_circulation.coords['lon']
		ny         = len(lat)
		nx         = len(lon)
		
		var_ana    = np.full((nday),np.nan)    
		ttt_ana    = np.full((nday),np.nan)
		cir_ana    = np.full((nday,ny,nx),np.nan)
		sample1    = np.zeros((nday,1),dtype=int)

		var_ana2   = np.full((nn),np.nan)   #nn次取样
		ttt_ana2   = np.full((nn),np.nan) 
		cir_ana2   = np.full((nn,ny,nx),np.nan)

		
		m=1
		var_uchronic = np.zeros((nn,1))
		for j in range(0,nn):
			print(f'{j}th sampling')
#
###--------先对日期做随机抽样（每nana(eg.20)个同类的日期挑一天,挑出nday(eg.7)个样本）------
			for i in range(0,nday):
				day2 = list(ana_day[i,:])
				sample1[i,:] = random.sample(day2,m)    #在day2中随机取1天，循环挑出nday天
			print(f'sample date is {sample1}')

			#不能用时间戳用这个！！
			sample2  = pd.PeriodIndex(sample1[:,0], dtype='period[D]', freq=None)        
			print(sample2)  

##-------找出nday个日期对应的变量--------

			t = 0
			for dt in sample2:    
				#print(dt)
				if dt in time:  
					#print(dt)
					#n  = time_list.index(dt)    #获取对应日期的下标，注：a.index()列表的用法（注：list只能找到第一个匹配的，不过这里时间都是唯一的
					#print(n)
					var_ana[t]     = self.data.loc[dt]
					cir_ana[t,:,:] = var_circulation.loc[dt]
					#print(self.data.loc[dt])
					if tmax_nodetrend is not None:
						ttt_ana[t]   =  tmax_nodetrend.loc[dt]
					t = t+1
				else:
					print('error!')
	
			var_ana2[j]       = np.nanmean(var_ana,axis=0)     ##nday天平均
			cir_ana2[j,:,:]   = np.nanmean(cir_ana,axis=0)
			if tmax_nodetrend is not None:
				ttt_ana2[j]   = np.nanmean(ttt_ana,axis=0)
		if tmax_nodetrend is not None:
			return var_ana2,cir_ana2,ttt_ana2
		else:
			return var_ana2,cir_ana2

	#=========================================================
	#同random_nn，但用于不能用时间戳(datetime64)的数据，picontrol
	#=========================================================
	def random2(self,start_year,end_year,nday,nn):  #nday事件的天数
		time 	 = self.data.coords['time']
		#lat  	 = self.data.coords['lat']
		#lon  	 = self.data.coords['lon']
		#ny       = len(lat)
		#nx       = len(lon)

		##-------先做7天的滑动平均，这样抽到1天就代表了7天的平均(or 3天)--------
		nyr = end_year-start_year+1
		j=0
		var_windows = np.full((nyr*(60+nday)),np.nan)


		#不能用loc选择时间所以用以下方法！
		data2       = self.data.data.reshape((nyr,123))
		for i in range(0,nyr):
			var_windows[j:j+60+nday]  = data2[i,27:94]   #5.28-8.02
			j = j +60+nday
			
		
		var_windows = var_windows.reshape((nyr,60+nday))   #因为是每年分别做滑动平均，所以reshape
		a           = np.repeat(1/nday, nday)
		var_w       = np.full((nyr,60+1),np.nan)
		#
		##卷积运算(滑动平均)
		for yr in range(0,nyr):
			b = np.convolve(var_windows[yr,:], a, mode='valid')  
			#print(b.shape)
			var_w[yr,:] = b
		var_w = var_w.reshape((nyr*(60+1)))
		#
		##----------把每年对应的事件前后30天的时间段先选出来------------ 
		#不能用时间戳的可以用pd.PeriodIndex!!
		year   = np.arange(start_year,end_year+1,1)
		time3  = list(pd.period_range(str(int(start_year))+'-05-31', str(int(start_year))+'-07-30', freq='D'))
		for y in year[1:nyr]:
			time2 = list(pd.period_range(str(int(y))+'-05-31',str(int(y))+'-07-30',freq='D')) 
			time3.extend(time2)
		time3  = pd.PeriodIndex(time3, dtype='period[D]', freq=None)
		print(time3[0:126])

		##注意这里对应也要修改！
		##-----------在前后30天中抽一天------------
		t=0
		var_random   = np.full((nn),np.nan)
		for j in range(0,nn):
			print(f'{j}th sampling')
			r_sample  = random.sample(list(time3),1)     #滑动平均了所以在前后30天抽一天即可
			r_sample2  = pd.PeriodIndex(r_sample, dtype='period[D]', freq=None)
			print(f'random sample date is {r_sample2}')
			#print(r_sample2)
			n  = list(time3).index(r_sample2)            #看这个时间在这61天中的索引
			print(time3[n]) 
			var_random[j] = var_w[n]
			#var_random[j,:,:] = random.sample(var_w,1)
			
		return var_random

    #=================================================
    #根据求得的日期，做取样，只给出对应的变量异常，不给空间场！
	def ana_random_tmax_ano(self,ana_day,nday,nn):         #ana_day挑选的环流相似的日期，ana_day[nday,nana],nday即极端事件的天数,nn-取样次数
		time 	   = self.data.coords['time']

		time2      = pd.to_datetime(time.data)
		print(time2)

		var_ana    = np.full((nday),np.nan)      

		sample1    = np.zeros((nday,1),dtype=int)
		var_ana2   = np.full((nn),np.nan)   #nn次取样
	
		m=1
		for j in range(0,nn):
			print(f'{j}th sampling')
#
###--------先对日期做随机抽样（每nana(eg.20)个同类的日期挑一天,挑出nday(eg.7)个样本）------
			for i in range(0,nday):
				day2 = list(ana_day[i,:])
				#print(day2)
				sample1[i,:] = random.sample(day2,m)    #在day2中随机取1天，循环挑出nday天
			#print(f'sample date is {sample1}')
			sample2 =[datetime.strptime(str(int(x)),'%Y%m%d') for x in sample1[:,0]]  #numpy转为时间格式
			sample2 = pd.DatetimeIndex(sample2, dtype='datetime64[ns]', freq=None)    #转为datetime64        
			print(sample2)  

##-------找出nday个日期对应的变量--------
			t = 0
			a = 0
			for dt in sample2:    
				#print(dt)
				if dt in time2:   #time or time2为var变量中的每一天
					#print(dt)
					time_list = list(time2)
					n  = time_list.index(dt)    #获取对应日期的下标，注：a.index()列表的用法（注：list只能找到第一个匹配的，不过这里时间都是唯一的
					#print(n)
					var_ana[t] = self.data[n]
					print(self.data[n])
					t = t+1				
				else:
					print('error!')
	
			var_ana2[j]   = np.nanmean(var_ana,axis=0)     ##nday天平均

		return var_ana2

	#=======================================================
    #根据求得的日期，做取样，只给出对应的土壤湿度异常
	def ana_random_sm_ano2(self,ana_day,nday,nn):         #ana_day挑选的环流相似的日期，ana_day[nday,nana],nday即极端事件的天数,nn-取样次数
		time 	   = self.data.coords['time']
		lat  	   = self.data.coords['lat']
		lon  	   = self.data.coords['lon']
		ny         = len(lat)
		nx         = len(lon)

		time2      = pd.to_datetime(time.data)
		print(time2)

		var_ana    = np.full((nday,ny,nx),np.nan)      

		sample1    = np.zeros((nday,1),dtype=int)
		var_ana2   = np.full((nn,ny,nx),np.nan)   #nn次取样
	
		m=1
		for j in range(0,nn):
			print(f'{j}th sampling')
#
###--------先对日期做随机抽样（每nana(eg.20)个同类的日期挑一天,挑出nday(eg.7)个样本）------
			for i in range(0,nday):
				day2 = list(ana_day[i,:])
				#print(day2)
				sample1[i,:] = random.sample(day2,m)    #在day2中随机取1天，循环挑出nday天
			#print(f'sample date is {sample1}')
			sample2 =[datetime.strptime(str(int(x)),'%Y%m%d') for x in sample1[:,0]]  #numpy转为时间格式
			sample2 = pd.DatetimeIndex(sample2, dtype='datetime64[ns]', freq=None)    #转为datetime64        
			#print(sample2)  

##-------找出nday个日期对应的变量--------
			t = 0
			a = 0
			for dt in sample2:    
				#print(dt)
				if dt in time2:   #time or time2为var变量中的每一天
					#print(dt)
					time_list = list(time2)
					n  = time_list.index(dt)    #获取对应日期的下标，注：a.index()列表的用法（注：list只能找到第一个匹配的，不过这里时间都是唯一的
					#print(n)
					var_ana[t,:,:] = self.data[n,:,:]
					t = t+1				
				else:
					print('error!')
	
			var_ana2[j,:,:]   = np.nanmean(var_ana,axis=0)     ##nday天平均

		return var_ana2

	##=======趋势线========
	def trend(x,y):
		z = np.polyfit(x, y, 1) #a,b,设置自由度為1
		print(f"z: {z}")
		# 生成的多項式對象(y = ax + b)
		p = np.poly1d(z)   #p= ax+b
		print(f"p: {p}")
		yy = p(x)  #拟合得到的y
		print(p[1])
		return yy,p[1] 
	def trend2(x,y):
		#排序，再回归
		#data     = {'A':x,'B':y}
		#unsorted = pd.DataFrame(data)
		#print(unsorted)
		#sortt  = unsorted.sort_values(by='A') #根据x排序
		#print(sortt)
		#xx = sortt.A
		#yy = sortt.B
		b,a,r,p,error = stats.linregress(x,y)
		yy           = b*x+a
		print(b)
		print(p)
		return yy,b,p

	#============================
	#计算大于某个数的概率
	def cdff(data,X):
		data2             = sorted(data)
		shape, loc, scale = gev.fit(data2)
	
		xx = np.linspace(data2[0]-1,data2[-1]+1, num=1000)
		#yy = gev.pdf(xx, shape, loc, scale)
		#print(xx)
		cdf = gev.cdf(xx,shape, loc, scale)
		#计算大于X的概率
		#n    = list(xx).index(round(X)) #X在xx中的索引
		n    = (np.abs(xx-X)).argmin()   #argmin表示使目标函数f(x)取最小值时的变量值
		#print(n)
		prob = 1 - cdf[n]
		#print(f'Probability greater than 2021 is {prob:.4f}')
	
		return prob


	#=================================
	#再取样！！！！！！！
	def scaleextreme(samples,ex):
		p  = cal_func.cdff(samples,ex)   #直接得到概率
		return p

	def book(var1,var2):
		scale=np.zeros((10000)) #再取样次数
		for i in range(10000):
			bootstrapsample = resample(var1,replace=1)
			tt              = cal_func.scaleextreme(bootstrapsample,var2)
			scale[i]        = tt
		
		pro_95     = stats.scoreatpercentile(scale, 95)
		pro_5      = stats.scoreatpercentile(scale, 5)
		pro        = np.nanmedian(scale)
		print(f'median is {pro}({pro_5}-{pro_95})')
		return pro

	#-----FAR&PR----
	def book2(var1,var2,var3,type):
		scale=np.zeros((10000))
		for i in range(10000):
			bootstrapsample1 = resample(var1,replace=1)
			bootstrapsample2 = resample(var2,replace=1)
			if type=='PR':
				p1 = cal_func.cdff(bootstrapsample1,var3)
				p2 = cal_func.cdff(bootstrapsample2,var3)
				tt = p1/p2
			else:
				p1 = cal_func.cdff(bootstrapsample1,var3)
				p2 = cal_func.cdff(bootstrapsample2,var3)
				tt = 1-p1/p2
			scale[i]         = tt
		
		pro_95     = stats.scoreatpercentile(scale, 95)
		pro_5      = stats.scoreatpercentile(scale, 5)
		pro        = np.nanmedian(scale)
	
		print(type)
		print(f'median is {pro}({pro_5}-{pro_95})')
		return pro

#=====================================================================================================
#*************相似的环流下，取出前30天的土壤湿度，平均，分别取7天，再平均，得到对应的土壤湿度*************
#=====================================================================================================
	def flow_analogue_sm(self,ana_day,nday,nn):         #ana_day挑选的环流相似的日期，ana_day[nday,nana],nday即极端事件的天数,nn-取样次数
		time 	   = self.data.coords['time']
		lat  	   = self.data.coords['lat']
		lon  	   = self.data.coords['lon']
		ny         = len(lat)
		nx         = len(lon)
		#datetimeindex = self.data.indexes['time'].to_datetimeindex()
		#self.data.coords['time'] = datetimeindex
		time2      = pd.to_datetime(time.data)  #土壤湿度的时间
		print(time2)
		#print(ok)

		var_ana    = np.full((nday,ny,nx),np.nan)    
		sample1    = np.zeros((nday,1),dtype=int)
		var_ana2   = np.full((nn,ny,nx),np.nan)   #nn次取样

		m=1
		for j in range(0,nn):
			print(f'{j}th sampling')
#
###--------先对日期做随机抽样（每nana(eg.20)个同类的日期挑一天,挑出nday(eg.7)个样本）------
			for i in range(0,nday):
				day2 = list(ana_day[i,:])
				#print(day2)
				sample1[i,:] = random.sample(day2,m)    #在day2中随机取1天，循环挑出nday天
			print(f'sample date is {sample1}')
			sample2 =[datetime.strptime(str(int(x)),'%Y%m%d') for x in sample1[:,0]]  #numpy转为时间格式
			sample2 = pd.DatetimeIndex(sample2, dtype='datetime64[ns]', freq=None)    #转为datetime64        
			#print(sample2)  

##-------找出nday个日期对应的变量--------
			t = 0
			a = 0
			for dt in sample2:    
				#print(dt)
				if dt in time2:   #time or time2为var变量中的每一天
					print(dt)
					time_list = list(time2)
					n  = time_list.index(dt)    #获取对应日期的下标，注：a.index()列表的用法（注：list只能找到第一个匹配的，不过这里时间都是唯一的
					#print(n)
					var_ana[t,:,:] = np.nanmean(self.data[n-30:n,:,:],axis=0)  #前30天的土壤湿度的平均
					#print(self.data[n-30:n,:,:])
					#var_ana[t,:,:] = self.data[n,:,:]
					print(self.data[n,:,:])
					t = t+1				
				else:
					print('error!')
	
			var_ana2[j,:,:]   = np.nanmean(var_ana,axis=0)     ##nday天平均

		return var_ana2

#======================================
#随机取样,随机取30天的土壤湿度
#======================================
	def random_sm(self,start_year,end_year,nday,nn,day1,day2,day3,day4):  #nday事件的天数
		time 	 = self.data.coords['time']
		lat  	 = self.data.coords['lat']
		lon  	 = self.data.coords['lon']
		ny       = len(lat)
		nx       = len(lon)

	    #-------选取土壤湿度的窗口日期，并做滑动平均------------
		nyr = end_year-start_year+1
		var_windows = np.full((nyr,66+30,ny,nx),np.nan)
		#事件是6.27-7.3，而环流相似法这里给的是前后30天的窗口，也就是5.28-8.2，因为我们抽取土壤湿度是取事件的前30天，
		#所以抽取土壤湿度的窗口应该是4.28-8.1
		for i in range(start_year,end_year+1):
			if int(day1[0:2])>int(day2[0:2]):   #月份比较，跨年了，所以要i+1
				print('such as DJF')
				var_windows[i-start_year,:,:,:] = self.data.loc[str(i)+'-'+day1:str(i+1)+'-'+day2]   
			else:
				var_windows[i-start_year,:,:,:] = self.data.loc[str(i)+'-'+day1:str(i)+'-'+day2]
		
		#做30天的滑动平均
		a           = np.repeat(1/nday, nday)
		var_w       = np.full((nyr,67,ny,nx),np.nan)
		#
		##卷积运算(滑动平均)
		for yr in range(0,nyr):
			for i in range(0,ny):
				for j in range(0,nx):
					b = np.convolve(var_windows[yr,:,i,j], a, mode='valid')  
					var_w[yr,:,i,j] = b
		var_w = var_w.reshape((nyr*67,ny,nx))
		
		##----------把滑动后的窗口的时间抽取出来------------
		#可以转化为pandas的datetime64即时间戳用以下程序!!
		j=0
		time3 = np.zeros((nyr*67))
		for i in range(start_year,end_year+1):
			if int(day3[0:2])>int(day4[0:2]):   #月份比较，跨年了，所以要i+1
				time3[j:j+67]  = time.loc[str(i)+'-'+day3:str(i+1)+'-'+day4]
			else:
				time3[j:j+67]  = time.loc[str(i)+'-'+day3:str(i)+'-'+day4]
			j = j+67
		time3 = pd.to_datetime(time3)  

		##-----------抽一天的土壤湿度，代表了任意选取的30天的土壤湿度------------
		t=0
		var_random   = np.full((nn,ny,nx),np.nan)
		for j in range(0,nn):
			print(f'{j}th sampling')
			r_sample  = random.sample(list(time3),1)     
			r_sample2 = pd.to_datetime(r_sample)
			print(f'random sample date is {r_sample2}')
			#print(r_sample2)
			n  = list(time3).index(r_sample2)            #看这个时间在这滑动后时间中的索引
			print(time3[n]) 
			var_random[j,:,:] = var_w[n,:,:]
			
		return var_random

#-----对土壤湿度直接应用环流相似法，挑选了与事件的前30天土壤湿度类似的天数----
#	def sm_ana_random(self,ana_day,nday,tmax,nn):         #ana_day挑选的环流相似的日期，ana_day[nday,nana],nday即极端事件的天数,nn-取样次数
#		time 	   = self.data.coords['time']
#		lat  	   = self.data.coords['lat']
#		lon  	   = self.data.coords['lon']
#		ny         = len(lat)
#		nx         = len(lon)
#		#datetimeindex = self.data.indexes['time'].to_datetimeindex()
#		#self.data.coords['time'] = datetimeindex
#		time2      = pd.to_datetime(time.data)  #土壤湿度的日期4-12月
#		print(time2)
#		
#		var_ana    = np.full((nday,ny,nx),np.nan)    
#		ta_ana     = np.full((nday,ny,nx),np.nan)
#		sample1    = np.zeros((nday,1),dtype=int)
#		var_ana2   = np.full((nn,ny,nx),np.nan)   #nn次取样
#		ta_ana2    = np.full((nn,ny,nx),np.nan)
#
#		
#		m=1
#		for j in range(0,nn):
#			print(f'{j}th sampling')
##
####--------先对日期做随机抽样（每nana(eg.20)个同类的日期挑一天,挑出nday(eg.7)个样本）------
#			for i in range(0,nday):
#				day2 = list(ana_day[i,:])
#				#print(day2)
#				sample1[i,:] = random.sample(day2,m)    #在day2中随机取1天，循环挑出nday天
#			print(f'sample date is {sample1}')
#			sample2 =[datetime.strptime(str(int(x)),'%Y%m%d') for x in sample1[:,0]]  #numpy转为时间格式
#			sample2 = pd.DatetimeIndex(sample2, dtype='datetime64[ns]', freq=None)    #转为datetime64        
#			print(sample2)  
#
###-------找出nday个日期对应的变量--------
#			t = 0
#			a = 0
#			for dt in sample2:    
#				#print(dt)
#				print(f'now is the {t}')
#				if dt in time2:   #time or time2为var变量中的每一天
#					#print(dt)
#					time_list = list(time2)
#					n  = time_list.index(dt)    #获取对应日期的下标(土壤湿度对应的时间)，注：a.index()列表的用法（注：list只能找到第一个匹配的，不过这里时间都是唯一的
#					#print(n)
#					nyr= (n+1)//153
#					nd = (n+1)%153      #取余，也就是在当年的第nd天
#					#print(f'{nyr},{nd}')
#					n2 = nyr*123+(nd-30)-1   #
#					#print(n2)
#					var_ana[t,:,:] = self.data[n,:,:] #土壤湿度,注意土壤湿度也要取和温度一样的时间，否则时间会对不上！！
#					ta_ana[t,:,:]  = np.nanmean(tmax[n2+(30-t):n2+(30-t+7)],axis=0)      #温度异常取土壤湿度的后n天
#					print(self.data[n].coords['time'])
#					print(tmax[n2+(30-t):n2+(30-t+7)].coords['time'])
#					t = t+1				
#				else:
#					print('error!')
#	
#			var_ana2[j,:,:]   = np.nanmean(var_ana,axis=0)     ##nday天平均
#			ta_ana2[j,:,:]    = np.nanmean(ta_ana,axis=0)
#
#		return var_ana2,ta_ana2

#---去除三次样条趋势----
	def detrend_fit(tmax_ano,yy=None):
		tmax_ano_year = tmax_ano.groupby('time.year').mean('time') #(year)
		year          = tmax_ano_year.coords['year']
		ttt           = np.linspace(year[0].data,year[-1].data,tmax_ano.data.shape[0])
		#print(ttt)
		y  = csaps(year,tmax_ano_year,ttt,smooth=0.0001,axis=0) #得到拟合的线,和原先数据一样的维度
		#print(y)
		tmax_ano2     = tmax_ano - y                            #去趋势
		#附属性
		#tmax_ano2     = cal_func(tmax_ano).attr(tmax_ano2)
		#print(tmax_ano2)
		if yy is not None:
			return tmax_ano2, y
		else:
			return tmax_ano2

#----去除线性趋势----
	def detrend_fit2(xdata):
		if xdata.ndim==3:
			#ny = xdata.data.shape[1]
			#nx = xdata.data.shape[2]
			time  = xdata.coords['time']
			#xdata.data[np.isnan(xdata.data)] =0
			x     = np.arange(1,len(time)+1)
			x     = xr.DataArray(x,\
		               dims=('time'),\
		               coords={'time':np.arange(1,len(time)+1),})
			b,p,interp = cal_function_from_jiang.regression_pcs_to_original_field_Xarray(x,xdata)
			y          = x*b+interp  #(time,lat,lon)
			#print(y.shape)
			#print(xdata)
			xdata_detrend = xdata.data - y
			print(xdata_detrend.shape)
			#xdata_detrend.data[xdata_detrend.data==0]=np.nan
			xdata_detrend  = xr.DataArray(xdata_detrend,\
			                 dims=('time','lat','lon'),\
			                 coords={'time':xdata.coords['time'],'lat':xdata.coords['lat'],'lon':xdata.coords['lon']})
			#y  = xr.DataArray(y,\
			#                 dims=('time','lat','lon'),\
			#                 coords={'time':xdata.coords['time'],'lat':xdata.coords['lat'],'lon':xdata.coords['lon']})
		if xdata.ndim==4:
			nm    = xdata.data.shape[0]
			time  = xdata.coords['time']
			xdata_detrend = np.full((xdata.data.shape),np.nan)

			x     = np.arange(1,len(time)+1)
			x     = xr.DataArray(x,\
		               dims=('time'),\
		               coords={'time':np.arange(1,len(time)+1),})
			for i in range(0,nm):
				print(f'model{i+1}')
				b,p,interp = cal_function_from_jiang.regression_pcs_to_original_field_Xarray(x,xdata[i])
				y          = x*b+interp  #(time,lat,lon)
				xdata_detrend[i] = xdata[i].data - y
			xdata_detrend  = xr.DataArray(xdata_detrend,\
		               dims=('member','time','lat','lon'),\
		               coords={'member':xdata.coords['member'],'time':xdata.coords['time'],'lat':xdata.coords['lat'],'lon':xdata.coords['lon']})
		return xdata_detrend

#-------------------------------------------
#Assign the attribution of A to B
#-------------------------------------------
	def attr(self, B):
		if not isinstance(self.data, xr.DataArray):
			raise  TypeError('Data must be xarray DataArray.')
		if self.data.ndim==3:
			#print(self.data.ndim)
			lat = self.data.coords['lat']
			lon = self.data.coords['lon']
			time= self.data.coords['time']

			B  = xr.DataArray(B,\
				        dims=('time','lat','lon'),\
				        coords={'time':time,'lat':lat,'lon':lon})
		return B

	def attr2(self,time,lat,lon):
		B  = xr.DataArray(self.data,\
				        dims=('time','lat','lon'),\
				        coords={'time':time,'lat':lat,'lon':lon})
		return B

	def  save_data_t_lat_lon(self,name,time,lat,lon,filename):
		self.data = xr.DataArray(self.data,\
			       	 dims=('time','lat','lon'),\
			        coords={'time':time,'lat':lat,'lon':lon})
		ds = xr.Dataset({name:self.data})   
		outputfile=filename
		ds.to_netcdf(outputfile, engine='netcdf4')


			










