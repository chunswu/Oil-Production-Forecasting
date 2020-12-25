from functions import *
import pyspark as ps
from pyspark.sql.types import *
from pyspark.sql.functions import struct, col, when, lit
from pyspark.sql import functions as f
from pandas_profiling import ProfileReport
import pandas as pd
import pickle

def clean_data(data):
    '''Cleans the fluid types from a spark DataFrame

    Parameters
    ----------
    data: DataFrame in spark
    
    Returns
    -------
    data: DataFrame in spark
    '''
    data = drop_na_column(data, ["fluid_type1"])

    wrong_fluid = ['HYBRID|X-LINK', 'X-LINK', 'ACID|OTHER FLUID', 
                   'OTHER FLUID|WATER', 'HYBRID|LINEAR GEL', 'HYBRID|SLICKWATER', 
                   'X-LINK|SLICKWATER', 'ACID|X-LINK', 'GEL|LINEAR GEL']

    right_fluid = ['HYBRID', 'GEL', 'ACID', 
                   'WATER', 'HYBRID', 'HYBRID', 
                   'HYBRID', 'HYBRID', 'GEL']

    data = fix_fluid_types(data, wrong_fluid, right_fluid)
    data = fill_fluid_na(data)
    data = data.distinct()

    combine_fluids = ['HYBRID', 'SLICKWATER', 'GEL']

    for fluid in combine_fluids:
        data = clean_fluid_type(data, fluid)
    
    columns_to_drop = ['hybrid1', 'hybrid2', 'hybrid3', 'hybrid4', 'hybrid5',
                   'slickwater1', 'slickwater2', 'slickwater3', 'slickwater4', 'slickwater5',
                   'gel1', 'gel2', 'gel3', 'gel4', 'gel5',
                   'FluidVol1', 'fluid_type1','FluidVol2','fluid_type2', 'FluidVol3', 
                   'fluid_type3', 'FluidVol4', 'fluid_type4', 'FluidVol5', 'fluid_type5']
    data = data.drop(*columns_to_drop)
    return data

def finished_form(data1, data2):
    '''join two spark DataFrames to make final data set

    Parameters
    ----------
    data1: DataFrame in spark
    data2: DataFrame in spark
    
    Returns
    -------
    data: DataFrame in pandas
    '''
    data2 = data2.na.replace({104.8865041: -104.8865041})
    join_data = data1.join(data2, ['api'], 'left_outer')

    join_data = join_data.filter((join_data.formation != 'GREENHORN') & 
                                 (join_data.formation != 'SUSSEX'))
    join_data = join_data.withColumnRenamed('hybrid_collect', 'Hybrid')
    join_data = join_data.withColumnRenamed('slickwater_collect', 'Slickwater')
    join_data = join_data.withColumnRenamed('gel_collect', 'Gel')
    # join_data.show()

    # data = join_data.toPandas()
    # data = data[data.formation != 'GREENHORN']
    # data = data[data.formation != 'SUSSEX']
    # data = data.rename({'hybrid_collect': 'Hybrid',
    #                     'slickwater_collect': 'Slickwater', 
    #                     'gel_collect': 'Gel'}, axis=1)
    return join_data

def column_expand(data, old_column, new_column):
    '''Takes in Spark DataFrame and column name, create new column with 1 or 0 as values

    Parameters
    ----------
    data: DataFrame in Spark
    old_column: string
    new_column: string

    Returns
    -------
    data: DataFrame in padas
    '''
    # data.withColumn(new_column, f.when(f.col(old_column) == new_column, 1).otherwise(0))
    # data[new_column] = np.where(data[old_column] == new_column, 1, 0)
    return data.withColumn(new_column, f.when(f.col(old_column) == new_column, 1).otherwise(0))


if __name__ == '__main__':
    spark = (ps.sql.SparkSession.builder 
        .master("local[4]") 
        .appName("sparkSQL exercise") 
        .getOrCreate()
        )
    sc = spark.sparkContext

    df = spark.read.csv('../data/dj_basin.csv',
                         header=True,
                         quote='"',
                         sep=",",
                         inferSchema=True)
    df.createOrReplaceTempView("data")

    fluid_data = spark.sql("""
                    SELECT
                        api,
                        State,
                        FluidVol1,
                        UPPER(FluidType1) AS fluid_type1,
                        FluidVol2,
                        UPPER(FluidType2) AS fluid_type2,
                        FluidVol3,
                        UPPER(FluidType3) AS fluid_type3,
                        FluidVol4,
                        UPPER(FluidType4) AS fluid_type4,
                        FluidVol5,
                        UPPER(FluidType5) AS fluid_type5
                    FROM data
                    """)

    parameter_data = spark.sql("""
                        SELECT 
                            api,
                            Latitude, 
                            Longitude,
                            UPPER(formation) AS formation,
                            LateralLength,
                            Azimuth,
                            TotalProppant AS Proppant,
                            Prod180DayOil AS day180
                        FROM data
                        """)

                            # Prod365DayOil AS day365,
                            # Prod545DayOil AS day545,
                            # Prod730DayOil AS day730,
                            # Prod1095DayOil AS day1095,
                            # Prod1460DayOil AS day1460,
                            # Prod1825DayOil AS day1825,

    fluid_data = clean_data(fluid_data)
    final_set = finished_form(fluid_data, parameter_data)

    formation_seperate = ['NIOBRARA', 'CODELL']
    state_seperate = ['COLORADO']

    for layers in formation_seperate:
        final_set = column_expand(final_set, 'formation', layers)

    for state in state_seperate:
        final_set = column_expand(final_set, 'State', state)

    columns_to_drop = ['formation', 'State']
    final_set = final_set.drop(*columns_to_drop)
    # final_set = final_set.drop(columns=['formation'])
    # final_set = final_set.drop(columns=['State'])
    final_set = final_set.dropna()
    order_columns = ['api', 'Latitude', 'Longitude', 
                     'LateralLength', 'Azimuth', 'Proppant',
                     'NIOBRARA', 'CODELL', 'COLORADO', 
                     'Hybrid', 'Slickwater', 'Gel', 'day180']
    final_set = final_set.select(order_columns)
    # final_set = final_set.set_index('api')
    # final_set.rename(columns={'TotalProppant': 'Total Proppant'}, inplace=True)
    # print(type(final_set))
    final_set.show()
    final_set.write.parquet("../model/data.parquet")
    # with open('../model/data.pkl', 'wb') as data_file:
    #     pickle.dump(final_set, data_file)

    # fluid_data = clean_data(fluid_data)
    # final_set = finished_form(fluid_data, parameter_data)
    # eda_report = ProfileReport(final_set)
    # eda_report.to_file(output_file='../html/clean_report.html')
    # final_data = fluid_data.join(parameter_data, ['api'], 'left_outer')

