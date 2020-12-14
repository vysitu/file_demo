# import libs
import numpy as np
import pandas as pd
import datetime as dt
# from statsmodels.tsa.seasonal import seasonal_decompose
# import scipy.stats as stats
import netCDF4 as nc
from IPython.display import clear_output

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import cartopy as cart

# <codecell> set plotting parameters
plt.rcParams["figure.figsize"] = (12,7) #(12,7) (16,10)
plt.rcParams.update({'font.size': 24})
plt.rcParams['figure.facecolor'] = 'white'

# <codecell> read data
threshold = -100

thing = "H2O_162"
#thing = 'NO2'
with nc.Dataset('/home/ysitu/ace-fts/ACEFTS_L2_v4p0_'+thing+'.nc','r') as dat:
#with nc.Dataset('/home/ysitu/NO2/ace/ACEFTS_L2_v4p0_NO2.nc','r') as dat:
    hdo1 = dat.variables[thing][:]
    lats1 = dat.variables['latitude'][:]
    lons1 = dat.variables['longitude'][:]
    alts1 = dat.variables['altitude'][:]
    pressure1 = dat.variables['pressure'][:]
    year1 = dat.variables['year'][:]
    month1 = dat.variables['month'][:]
    day1 = dat.variables['day'][:]
    hour1 = dat.variables['hour'][:]
hdo = np.where(hdo1<=threshold, np.nan, hdo1)

thing = "H2O"
#thing = 'NO2'
with nc.Dataset('/home/ysitu/ace-fts/ACEFTS_L2_v4p0_'+thing+'.nc','r') as dat:
#with nc.Dataset('/home/ysitu/NO2/ace/ACEFTS_L2_v4p0_NO2.nc','r') as dat:
    h2o1 = dat.variables[thing][:]
    lats2 = dat.variables['latitude'][:]
    lons2 = dat.variables['longitude'][:]
    alts2 = dat.variables['altitude'][:]
    pressure2 = dat.variables['pressure'][:]
    year2 = dat.variables['year'][:]
    month2 = dat.variables['month'][:]
    day2 = dat.variables['day'][:]
    hour2 = dat.variables['hour'][:]
h2o = np.where(h2o1<=threshold, np.nan, h2o1)

# convert to regular lonlat range -180~180
lons1 = np.where(lons1<-180, lons1+180, lons1)
lons1 = np.where(lons1>180, lons1-180, lons1)
lons2 = np.where(lons2<-180, lons2+180, lons2)
lons2  = np.where(lons2>180, lons2-180, lons2)

# <codecell> get parameters

lonBinSize = input('input Longitude grid size, default 5:   ' or 5)
latBinSize = input('input Latitude grid size, default 4:   ' or 5)
quant = input('input quantile limit to emit extreme values, default 0.01:  ' or 0.01)  #quantile limit

# <codecell> yearly parameters
latBin = np.arange(-90, 90, latBinSize)    #latitude bins
lonBin = np.arange(-180, 180, lonBinSize)  #longitude bins
hdoLayer = np.zeros((1, lonBin.shape[0], latBin.shape[0]))   #data receiver
h2oLayer = np.zeros((1, lonBin.shape[0], latBin.shape[0]))

startDate =  year1[0].astype(int).astype(str)+'-'+ month1[0].astype(int).astype(str)+'-1'
endDate =  year1[-1].astype(int).astype(str)+'-'+ month1[-1].astype(int).astype(str)+'-1'
dateList = pd.date_range(start =startDate, end = endDate, freq = 'M')
dateList2 =[]   #for plotting

xMesh, yMesh = np.meshgrid(lonBin, latBin)  #for plotting
obsCount1 = np.zeros((1, lonBin.shape[0], latBin.shape[0]))
obsCount1f = np.zeros((1, lonBin.shape[0], latBin.shape[0]))

for yr in np.unique(year1):   #only make yearly 
    condY1 = (year1 == yr)
    condY2 = (year2 == yr)
    pixel1 = []  # use to define layer pixels
    pixel2 = []  # attach row to receiver later
    layer1 = np.zeros((lonBin.shape[0], latBin.shape[0]))   #each layer
    layer2 = np.zeros((lonBin.shape[0], latBin.shape[0]))
    
    rowCount = int(0)   #each row(lat) +1
    for latMin in latBin:
        latMax = latMin+latBinSize
        colCount = int(0)   #each row(lat) reset, each col + 1
        for lonMin in lonBin:
            lonMax = lonMin + lonBinSize
            subset1 = dat1[(lats1>=latMin) & (lats1<latMax) & (lons1>=lonMin) & (lons1<lonMax) & condY1 ]
            subset2 = dat2[(lats2>=latMin) & (lats2<latMax) & (lons2>=lonMin) & (lons2<lonMax) & condY2 ]
            subset1f = subset1[(subset1>np.nanquantile(subset1, quant)) & (subset1<np.nanquantile(subset1, 1-quant))]
            subset2f = subset2[(subset2>np.nanquantile(subset2, quant)) & (subset2<np.nanquantile(subset2, 1-quant))]
            pixel1= (np.nanmean(subset1f))
            pixel2= (np.nanmean(subset2f))
            layer1[colCount, rowCount] = pixel1
            layer2[colCount, rowCount] = pixel2
            colCount = colCount+1   #for layer
        rowCount = rowCount+1   # for layer
    hdoLayer = np.concatenate([hdoLayer, layer1[np.newaxis, :, :]] , axis = 0)
    h2oLayer = np.concatenate([h2oLayer, layer2[np.newaxis, :, :]] , axis = 0)
    dateList2.append(str(int(yr)))

dhdo = (hdoLayer[1:, :, :]/h2oLayer[1:, :, :]) - 1
dhdo1 = dhdo

# <codecell> JFM
latBin = np.arange(-90, 90, latBinSize)    #latitude bins
lonBin = np.arange(-180, 180, lonBinSize)  #longitude bins
hdoTarget = np.zeros((1, lonBin.shape[0], latBin.shape[0]))   #DJF data receiver
h2oTarget = np.zeros((1, lonBin.shape[0], latBin.shape[0]))

startDate =  year1[0].astype(int).astype(str)+'-'+ month1[0].astype(int).astype(str)+'-1'
endDate =  year1[-1].astype(int).astype(str)+'-'+ month1[-1].astype(int).astype(str)+'-1'
dateList = pd.date_range(start =startDate, end = endDate, freq = 'M')
dateList2 =[]   #for plotting

xMesh, yMesh = np.meshgrid(lonBin, latBin)  #for plotting
for yr in np.unique(year1):   #only make yearly 
    condY1 = (year1 == yr)
    condY2 = (year2 == yr)
    condM1 = (month1 == 3) | (month1 == 1) | (month1 == 2)  # for JFM
    condM2 = (month2 == 3) | (month2 == 1) | (month2 == 2)
#     condM1 = (month1 == 6) | (month1 == 7) | (month1 == 8)  # for JJA
#     condM2 = (month2 == 6) | (month2 == 7) | (month2 == 8)  #for JJA
    pixel1 = []  # use to define layer pixels
    pixel2 = []  # attach row to receiver later
    layer1 = np.zeros((lonBin.shape[0], latBin.shape[0]))   #each layer
    layer2 = np.zeros((lonBin.shape[0], latBin.shape[0]))
    
    rowCount = int(0)   #each row(lat) +1
    for latMin in latBin:
        latMax = latMin+latBinSize
        colCount = int(0)   #each row(lat) reset, each col + 1
        for lonMin in lonBin:
            lonMax = lonMin + lonBinSize
            cond1 = (lats1>=latMin) & (lats1<latMax) & (lons1>=lonMin) & (lons1<lonMax)
            cond2 = (lats2>=latMin) & (lats2<latMax) & (lons2>=lonMin) & (lons2<lonMax)
            subset1 = dat1[cond1 & condY1 & condM1]
            subset2 = dat2[cond2 & condY2 & condM2]
            # vvv deal with outliers vvv
            subset1f = subset1[(subset1>np.nanquantile(subset1, quant)) & (subset1<np.nanquantile(subset1, 1-quant))] 
            subset2f = subset2[(subset2>np.nanquantile(subset2, quant)) & (subset2<np.nanquantile(subset2, 1-quant))]
            pixel1= (np.nanmean(subset1f))
            pixel2= (np.nanmean(subset2f))
            layer1[colCount, rowCount] = pixel1
            layer2[colCount, rowCount] = pixel2
            colCount = colCount+1   #for layer
        rowCount = rowCount+1   # for layer
    hdoTarget = np.concatenate([hdoTarget, layer1[np.newaxis, :, :]] , axis = 0)
    h2oTarget = np.concatenate([h2oTarget, layer2[np.newaxis, :, :]] , axis = 0)
    dateList2.append(str(int(yr)))
hdoJFM = hdoTarget[1:, :, :]  #DJF data receiver
h2oJFM = h2oTarget[1:, :, :]

dhdoJFM = (hdoJFM/h2oJFM) -1

# <codecell> AO index
ao = pd.read_csv('./AO_index.txt', delim_whitespace = True, header = None)
ao.columns = ['year','month','index']
dates = []
for i in range(ao.shape[0]):
    dates.append(ao['year'].values.astype(str)[i] +'-'+ ao['month'].values.astype(str)[i])
ao['date'] = pd.to_datetime(dates)
ao04 = ao[ao['year']>=2004].copy()

# <codecell> remove mean
ao04Norm = ao04['index'].values - np.nanmean(ao04['index'].values)  #overall mean AO removed from the series
ao04['norm'] = ao04Norm

aoMean = []   #receive yearly avg AO index
for i in range(2004, 2021):
    ss = ao04[ao04['year']==i].copy()
    temp = np.nanmean(ss['norm'].values)
    aoMean.append(temp)

aoM = pd.DataFrame(aoMean)  #yearly AO index mean
aoM['year'] = np.arange(2004, 2021, 1)
aoM.columns= ['ao', 'year']

# <codecell> negative/ positive AO years' mean HDO
gt0 = aoM[aoM['ao'] >0]['year'].values - 2004
lt0 = aoM[aoM['ao'] <0]['year'].values - 2004

while True:
    option3 = input('enter 1 to choose full series, 2 to choose JFM months: ' or 1)
    if (option3 == 1):
        print('choosing full series')
        theTS = dhdo1
        plotName0 = 'full'
        break
    elif (option3 == 2):
        print('choosing JFM months')
        theTS = dhdoJFM
        plotName0 = 'JFM'
        break
    else: 
        print('no such option, choose again')
        continue

gt1 = theTS[gt0, :,: ]
lt1 = theTS[lt0, :,: ]

posYears = np.nanmean(gt1, axis = 0)
negYears = np.nanmean(lt1, axis = 0)
diff = posYears-negYears

# <codecell> plot 
fig = plt.figure(figsize=(12, 7)) 
gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[20, 1]) 
ax0 = plt.subplot(gs[0], projection=cart.crs.Orthographic(0,90))
ax1 = plt.subplot(gs[1])

steps = 500
target = negYears.T  #lon-lat

ax0.coastlines()  #添加海岸线PlateCarree()
ax0.add_feature(cart.feature.BORDERS)   #添加国境线
ax0.gridlines()
minv = np.nanquantile(target, 0.1)
maxv = np.nanquantile(target, 0.9)

cm1 = cm.bwr
target1 = np.where((target>=minv)&(target<=maxv), target, np.nan)
# item1 = ax0.contourf(xMesh, yMesh, target1, transform=cart.crs.PlateCarree(),  extent =[-180, 180, -90, 90], levels = steps
#                    , vmin =minv, vmax =maxv, cmap = cm.cividis, origin = 'lower')   #画位图xMesh, yMesh,levels = steps
item1 = ax0.imshow(target, transform=cart.crs.PlateCarree(),  extent =[-180, 180, -90, 90]#, levels = steps
                   , vmin =minv, vmax =maxv, cmap = cm1, origin = 'lower')   #画位图xMesh, yMesh,levels = steps
# extent 是LonMin, LonMax, LatMin, LatMax

# ax0.set_title('Negative AO years:[2004 2005 2009 2010\n'+
#              ' 2012 2013 2014 2016 2019]')
#ax0.set_xlabel('altitude: 15.5 km')
ax0.set_title('Pos AO dHDO - Neg AO dHDO\n threshold: '+str(quant*100)+'% of the grid')
norm = mpl.colors.Normalize(vmin=minv, vmax=maxv)
tk1 = list(np.linspace(minv,maxv,5))
cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cm1, norm=norm, orientation='vertical', spacing = 'uniform')


plotName = plotName0+'_'+str(quant)+'_'+str(lonBinSize)+'x'+str(latBinSize)

fig.savefig('./plot/ao_filters/ao_diff_'+plotName+'.png', bbox_inches = 'tight')