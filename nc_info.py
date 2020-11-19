# Read NetCDF
import netCDF4 as nc
import numpy as np
import pandas as pd
# Ask for file info
fname= input('input file name: ')

# Ask for options
optionList = '''
Please select one of the following:
1. print all variable info
2. print all variable names
3. print all variable names and dimensions
4. print'''
option = input('')


with nc.Dataset(fname, 'r') as dat:
    if option == '1':
        print(dat.variables)    # give a list of variables
    if option == '2':
        print(dat.variables.keys())
        