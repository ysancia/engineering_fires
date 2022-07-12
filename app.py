
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import pyspark
import numpy as np

from pyspark import SparkConf
from pyspark.sql import SparkSession

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.feature import VectorAssembler, MinMaxScaler

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

preds = [int(row["Prediction"]) for row in predictions.select("Prediction").collect()]
actual = [int(row["size_index"]) for row in predictions.select("size_index").collect()]
st.write("Model loaded")
st.write("Model accuracy: ")
st.write(accuracy)


fig, ax = plt.subplots()
ax.scatter(actual, preds, color='b', s=60, alpha=0.1)
#plt.plot([0,7], [0,7], color='r')

ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted category',fontsize=20)
#st.pyplot(fig)

features2 = ["LAT","LON","maxT","minT","precip"]

lat = st.number_input("Enter your Latitude: ", min_value=None, max_value=None)
lon = st.number_input("Enter your Longitude: ", min_value=None, max_value=None)
high = st.number_input("Enter your daily high temperature (in F):  ", min_value=None, max_value=None)
low = st.number_input("Enter your daily low temperature (in F): ", min_value=None, max_value=None)
rain = st.number_input("Enter your precipitation (in inches): ", min_value=None, max_value=None)

lat = np.round(lat,decimals=1)
lon = np.round(lon,decimals=1)
high = (high - 32) * 5/9
low = (low - 32) * 5/9
rain = (rain * 25.4) * 10

new_input = spark.createDataFrame([(1, lat, lon, high, low, rain)],
									["id","LAT","LON","maxT","minT","precip"])

st.show(df.show())

va = VectorAssembler(inputCols=features2, outputCol = "features")

if new_input[4] != 0.00:
	new_input = va.transform(new_input)
	new_pred = model.transform(new_input)
	st.show(new_pred)
