from functions import *
import plotly.express as px
import pyspark as ps
from pyspark.sql.types import *
from pyspark.sql.functions import struct, col, when, lit

def make_map(string_label, pd_df):
    '''Create a map where oil wells are located with oil production information
    Parameters
    ----------
    string_label: string
    pd_df: pandas DataFrame

    Returns
    -------
    None
    '''
    # maps_df.dropna(subset=[value], inplace=True)
    day_str = string_label.replace('day', '')
    save_str = '../html/map_' + day_str + '.html'

    fig = px.scatter_mapbox(pd_df, lat="Latitude", lon="Longitude",
                            hover_name="api", 
                            hover_data=["Latitude", "Longitude"],
                            color=value,
                            color_continuous_scale='turbo',
                            zoom=8,
                            labels={value:'Barrels of Oil at ' + day_str + ' Days'})
    fig.update_layout(mapbox_style='carto-positron',
                    margin={"r":0,"t":0,"l":0,"b":0}
                    )
    fig.write_html(save_str)



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

    coorid_df = spark.sql("""
                        SELECT 
                            api,
                            State,
                            Latitude, 
                            Longitude,
                            TotalProppant,
                            Prod180DayOil AS day180,
                            Prod365DayOil AS day365,
                            Prod545DayOil AS day545,
                            Prod730DayOil AS day730,
                            Prod1095DayOil AS day1095,
                            Prod1460DayOil AS day1460,
                            Prod1825DayOil AS day1825
                        FROM data
                        """)

    prod_lst = ['day180', 'day365', 'day545', 'day730', 'day1095', 'day1460', 'day1825']

    maps_df = coorid_df.toPandas()

    for value in prod_lst:
        maps_df.dropna(subset=[value], inplace=True)
        # coorid_df = coorid_df.na.drop(subset=[value])
        make_map(value, maps_df)
        # make_map(value, coorid_df)
