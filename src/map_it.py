from functions import *
import plotly.express as px


def make_map(string_label, pd_df):
    '''
    Parameters
    ----------
    string_label: string
    pd_df: pandas DataFrame

    Returns
    -------
    None
    '''

    maps_df.dropna(subset=[value], inplace=True)
    day_str = value.replace('day', '')
    save_str = '../html/map_' + day_str + '.html'

    fig = px.scatter_mapbox(maps_df, lat="Latitude", lon="Longitude",
                            hover_name="api", 
                            hover_data=["Latitude", "Longitude"],
                            color=value,
                            zoom=7,
                            labels={value:'Barrels of Oil at ' + day_str + ' Days'})
    fig.update_layout(mapbox_style="open-street-map",
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
        make_map(value, maps_df)
