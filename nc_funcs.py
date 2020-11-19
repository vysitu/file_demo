def getBoundInd(series, minB, maxB):
    '''if series is 0~360, automatically convert to -180~180'''
    if np.max(series) > 200:
        series = np.where(series>180, series-360, series)
        print('converting to -180~180 range')
    minInd = np.argmin(np.abs(series-minB))
    maxInd = np.argmin(np.abs(series-maxB))
    try:
        if (minInd> maxInd):
            return(maxInd, minInd)
        elif (minInd< maxInd):
            return(minInd, maxInd)
    except:
        print('Error')