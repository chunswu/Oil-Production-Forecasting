import pyspark as ps
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import sklearn.metrics

def transData(data):
    '''Combines all columns of Spark dataframe into a list columns

    Parameters
    ----------
    df: dataframe in sparks
    
    Returns
    -------
    returns a dataframe
    '''
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])

if __name__ == '__main__':
    spark = (ps.sql.SparkSession.builder 
        .master('local[4]') 
        .appName('spark randomforest') 
        .getOrCreate()
        )
    sc = spark.sparkContext
    df = spark.read.parquet('../model/data.parquet')
    df = df.drop('api')
    df.show()
    transformed = transData(df)

    featureIndexer = VectorIndexer(inputCol='features', 
                                   outputCol='indexedFeatures',
                                   maxCategories=4).fit(transformed)

    data = featureIndexer.transform(transformed) 
    (train_data, test_data) = data.randomSplit([0.7, 0.3], seed=88)    

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

    predictions.select('features', 'label', 'prediction').show(20, False)
    predictions.describe().show()
    evaluator = RegressionEvaluator(labelCol='label', 
                                    predictionCol='prediction', 
                                    metricName='rmse')
    rmse = evaluator.evaluate(predictions)
    print('Root Mean Squared Error (RMSE) on test data = %g' % rmse)

    y_true = predictions.select('label').toPandas()
    y_pred = predictions.select('prediction').toPandas()
    r2_score = sklearn.metrics.r2_score(y_true, y_pred)
    print('r2_score: {:4.3f}'.format(r2_score))
    print('FEATURES IMPORTANCES: ', model.stages[-1].featureImportances)
    model.stages[-1].trees
    rfModel = model.stages[1]
    print(rfModel)

    va = model.stages[-2]
    tree = model.stages[-1]
    display(tree)
    print(list(zip(va.getOutputCol(), tree.featureImportances)))