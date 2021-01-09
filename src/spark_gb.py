import pyspark as ps
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import sklearn.metrics
import numpy as np

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
    indexers = [StringIndexer(inputCol=c, outputCol='{0}_indexed'.format(c))
                 for c in categoricalCols]

    # default setting: dropLast=True
    encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(),
                outputCol='{0}_encoded'.format(indexer.getOutputCol()),dropLast=dropLast)
                for indexer in indexers]

    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]
                                + continuousCols, outputCol='features')

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
#    return Row(label=row['Sales'],
#               features=Vectors.dense([row['TV'],
#                                       row['Radio'],
#                                       row['Newspaper']]))
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])


if __name__ == '__main__':
    spark = (ps.sql.SparkSession.builder 
        .master('local[4]') 
        .appName('spark gradientboost') 
        .getOrCreate()
        )
    sc = spark.sparkContext
    df = spark.read.parquet('../model/data.parquet')
    # print('TYPE: ', type(df))
    df.show()
    transformed= transData(df)
    # transformed.show(5, False)
    featureIndexer = VectorIndexer(inputCol='features', 
                                   outputCol='indexedFeatures',
                                   maxCategories=4).fit(transformed)

    data = featureIndexer.transform(transformed) 
    # data.show(5, False)

    (train_data, test_data) = data.randomSplit([0.7, 0.3], seed=88) 
    r2_lst = []
    # range(0.01, 1.0, 0.01)
    # for tree in range(35, 105, 5): # np.linspace(0.05, 1, 20):
    gbt = GBTRegressor(featuresCol='indexedFeatures',
                        featureSubsetStrategy='auto',
                        stepSize=0.05,
                        maxDepth=5,
                        maxBins=95,
                        maxIter=150)
    # featureSubsetStrategy='auto'
    # stepSize=0.05
    # maxDepth=5
    # maxBins=95
    # maxIter=150
    # Root Mean Squared Error (RMSE) on test data = 7006.32
    # r2_score: 0.968
    pipeline = Pipeline(stages=[featureIndexer, gbt])
    model = pipeline.fit(train_data)

    predictions = model.transform(test_data)

    # Select example rows to display.
    predictions.select('features', 'label', 'prediction').show(100, False)

    evaluator = RegressionEvaluator(labelCol='label', 
                                    predictionCol='prediction', 
                                    metricName='rmse')
    rmse = evaluator.evaluate(predictions)
    print('Root Mean Squared Error (RMSE) on test data = %g' % rmse)

    
    y_true = predictions.select('label').toPandas()
    y_pred = predictions.select('prediction').toPandas()
    r2_score = sklearn.metrics.r2_score(y_true, y_pred)
    # r2_lst.append([tree, r2_score, rmse])
    print('r2_score: {:4.3f}'.format(r2_score))

    print('FEATURES IMPORTANCES: ', model.stages[-1].featureImportances)

    model.stages[-1].trees

    # print(r2_lst)

    rfModel = model.stages[1]
    print(rfModel)  # summary only