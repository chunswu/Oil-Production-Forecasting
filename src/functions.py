import numpy as np
from pyspark.sql.types import *
from pyspark.sql.functions import struct, col, when, lit

def drop_na_column(df, lst):
    '''Removes rows with null or n/a values from a dataframe

    Parameters
    ----------
    df: dataframe in sparks
    lst: list of strings
    
    Returns
    -------
    returns a dataframe
    '''
    return df.na.drop(subset=lst)

def fix_fluid_types(df, wrong_lst, right_lst):
    '''Renames the fluid types in a dataframe with the correct fluid type

    Parameters
    ----------
    df: dataframe in sparks
    wrong_lst: list of strings
    right_lst: list of strings
    
    Returns
    -------
    returns a dataframe
    '''    
    for i in range(1,6):
        df = df.na.replace(wrong_lst, right_lst, 'fluid_type'+str(i))
    return df

def fill_fluid_na(df):
    '''Replaces null or na in a column wiht blank or 0

    Parameters
    ----------
    df: dataframe in sparks
    
    Returns
    -------
    returns a dataframe 
    ''' 
    for i in range(1,6):
        df = df.na.fill({'fluid_type'+str(i): ''})
        df = df.na.fill({'FluidVol'+str(i): 0})
    
    return df

def clean_fluid_type(df, fluid_sys):
    '''Passed in spark DataFrame and strings of fluid systems and sums up the volumes
       across all the fluid types

    Parameters
    ----------
    df: Spark DataFrame
    fluid_sys: string name of fluid system
    
    Returns
    -------
    returns DataFrame
    '''
    fluid_vol = 'FluidVol'
    fluid_type = 'fluid_type'
    lowcase_fluid = fluid_sys.lower() + "_collect"
    low = 1
    high = 6

    df = df.withColumn(lowcase_fluid, lit(0))
    for i in range(low, high):
        df = df.withColumn(fluid_sys.lower()+str(i), when(col(fluid_type+str(i)) == fluid_sys, col(fluid_vol+str(i))).otherwise(0))
        df = df.withColumn(lowcase_fluid, col(lowcase_fluid) + col(fluid_sys.lower()+str(i)))

    return df
