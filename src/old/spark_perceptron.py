import pyspark as ps
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import udf, StringType
# from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
# from pyspark.ml.evaluation import RegressionEvaluator
import sklearn.metrics
import warnings
warnings.filterwarnings('ignore')

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
    df.show(10, False)
    transformed= transData(df)
    data = transformed
    data.show(10, False)

    splits = data.randomSplit([0.7, 0.2, 0.1], seed=88)
    train = splits[0]
    test = splits[1]

    mlpc = MultilayerPerceptronClassifier(labelCol='label',
                                          featuresCol='features',
                                          maxIter=10,
                                          layers=[13, 8, 1],
                                          blockSize=128,
                                          solver='gd',
                                          seed=88)
    paramGrid = ParamGridBuilder().build()
    # create evaluator
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")

    # create cross validation object
    crossval = CrossValidator(estimator=mlpc,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=10)  

    # run cross-validation, and choose the best set of parameters.
    cvModel = crossval.fit(train)
    # pipeline = Pipeline(stages=[data, perceptron])
    # model = mlpc.fit(train)

    # compute accuracy on the test set
    # result = model.transform(test)
    # predictionAndLabels = result.select("prediction", "label")
    # evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    # print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))