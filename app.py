
import streamlit as st
import pandas as pd
import matplotlib as plt
import pyspark

from pyspark import SparkConf
from pyspark.sql import SparkSession

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

conf = SparkConf().setAppName("model").setMaster("local")
spark = SparkSession.builder.config(conf=conf).getOrCreate()
#model = LogisticRegressionModel.load("./lr2")

df = spark.read.load("processed_combined.parquet")
train_df, test_df = df.randomSplit([0.7,0.3],seed=25)

model = LogisticRegression(labelCol = "size_index", featuresCol = "features", predictionCol = "Prediction")
model = model.fit(train_df)



predictions = model.transform(test_df).cache()
evaluate_acc = MulticlassClassificationEvaluator(labelCol="size_index",
											predictionCol="Prediction",
											metricName="accuracy")
accuracy = evaluate_acc.evaluate(predictions)
#evaluate_r2 = MulticlassClassificationEvaluator(labelCol="size_index",
											#predictionCol="Prediction",
											#metricName="r2")
#r2 = evaluate_r2.evaluate(predictions)
#evaluate_rmse = MulticlassClassificationEvaluator(labelCol="size_index",
											#predictionCol="Prediction",
											#metricName="accuracy")
#rmse = evaluate_rmse.evaluate(predictions)

preds = [int(row["prediction"]) for row in predictions.select("Prediction").collect()]
actual = [int(row["size_index"]) for row in predictions.select("size_index").collect()]

fig, ax = plt.subplots()
ax.scatter(actual, preds, color='b', s=60, alpha=0.1)
plt.plot([5,250], [5,250], color='r')
plt.xlim([0, 260])
plt.ylim([0, 260])
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted Prices',fontsize=20)
col5.pyplot(fig)