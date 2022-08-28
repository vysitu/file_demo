from osgeo import gdao, osr

def export_tif(write_path, arr, ref_geotrans, nodata_value=-9999):
    ref_width = arr.shape[1]
    ref_height = arr.shape[0]
    compress = ["COMPRESS=LZW"] 
    datatype = gdal.GDT_Float32 
    out_ds = gdal.GetDriverByName("GTiff").Create(
        write_path, ref_width, ref_height, 1, datatype, compress
    )
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_ds.SetProjection(srs.ExportToWkt())
    out_ds.SetGeoTransform(ref_geotrans)
    out_ds.GetRasterBand(1).WriteArray(arr)
    out_ds.GetRasterBand(1).SetNoDataValue(nodata_value)
    out_ds.FlushCache()
    out_ds = None
    
    return write_path