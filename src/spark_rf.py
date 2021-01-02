import pyspark as ps
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.regression import RandomForestRegressor
 
import sklearn.metrics

def get_dummy(df,indexCol,categoricalCols,continuousCols,labelCol,dropLast=False):
    '''Get dummy variables and concat with continuous variables for ml modeling

    Parameters
    ----------
    df: the dataframe
    categoricalCols: the name list of the categorical data
    continuousCols:  the name list of the numerical data
    labelCol:  the name of label column
    dropLast:  the flag of drop last column
    
    Returns
    -------
    features matrix
    '''
    indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
                 for c in categoricalCols]

    # default setting: dropLast=True
    encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(),
                outputCol="{0}_encoded".format(indexer.getOutputCol()),dropLast=dropLast)
                for indexer in indexers]

    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]
                                + continuousCols, outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler])

    model=pipeline.fit(df)
    data = model.transform(df)

    if indexCol and labelCol:
        # for supervised learning
        data = data.withColumn('label',col(labelCol))
        return data.select(indexCol,'features','label')
    elif not indexCol and labelCol:
        # for supervised learning
        data = data.withColumn('label',col(labelCol))
        return data.select('features','label')
    elif indexCol and not labelCol:
        # for unsupervised learning
        return data.select(indexCol,'features')
    elif not indexCol and not labelCol:
        # for unsupervised learning
        return data.select('features')

# convert the data to dense vector
#def transData(row):
#    return Row(label=row["Sales"],
#               features=Vectors.dense([row["TV"],
#                                       row["Radio"],
#                                       row["Newspaper"]]))
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])

if __name__ == '__main__':
    spark = (ps.sql.SparkSession.builder 
        .master("local[4]") 
        .appName("spark randomforest") 
        .getOrCreate()
        )
    sc = spark.sparkContext
    df = spark.read.parquet('../model/data.parquet')
    # print('TYPE: ', type(df))
    df.show()
    transformed= transData(df)
    # transformed.show(5)

    featureIndexer = VectorIndexer(inputCol="features", 
                                   outputCol="indexedFeatures",
                                   maxCategories=4).fit(transformed)

    data = featureIndexer.transform(transformed) 
    # data.show(5, True)

    (train_data, test_data) = data.randomSplit([0.7, 0.3], seed=88)    
    # print('*********************** TRAIN DATA ***********************')
    # train_data.show(5)
    # print('*********************** TEST DATA ***********************')
    # test_data.show(5)
    # r2_lst = []
    # strategy = ['auto', 'onethird', 'sqrt', 'log2']
    # for tree in range(13, 30, 1):
    rf = RandomForestRegressor(featuresCol='features',
                               labelCol='label',
                               numTrees=150, 
                               maxDepth=12,
                               maxBins=25,
                               featureSubsetStrategy='auto', 
                               seed=88)
    # numTrees=150
    # maxDepth=12
    # maxBins=25
    # featuresSubsetStrategy='auto'
    # r2_score = 0.953
    # rmse 8471.48
    pipeline = Pipeline(stages=[featureIndexer, rf])
    model = pipeline.fit(train_data)

    predictions = model.transform(test_data)

    # Select example rows to display.
    predictions.select("features","label", "prediction").show(20, False)

    evaluator = RegressionEvaluator(labelCol="label", 
                                    predictionCol="prediction", 
                                    metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    
    y_true = predictions.select("label").toPandas()
    y_pred = predictions.select("prediction").toPandas()
    r2_score = sklearn.metrics.r2_score(y_true, y_pred)
    # r2_lst.append([tree, r2_score, rmse])
    print('r2_score: {:4.3f}'.format(r2_score))

    model.stages[-1].featureImportances

    model.stages[-1].trees

    # print(r2_lst)

    rfModel = model.stages[1]
    print(rfModel)  # summary only