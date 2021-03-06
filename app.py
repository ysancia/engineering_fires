
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import pyspark
import numpy as np

from pyspark import SparkConf
from pyspark.sql import SparkSession

from pyspark.sql import types as T

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
st.set_page_config(layout="wide")



fig, ax = plt.subplots()
ax.scatter(actual, preds, color='b', s=60, alpha=0.1)
#plt.plot([0,7], [0,7], color='r')

ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted category',fontsize=20)
#st.pyplot(fig)

features2 = ["LAT","LON","maxT","minT","precip"]



with st.sidebar:
	st.write("Model loaded")
	st.write("Model accuracy: ")
	st.write(accuracy)
	lat = st.number_input("Enter your Latitude: ", min_value=None, max_value=None)
	lon = st.number_input("Enter your Longitude: ", min_value=None, max_value=None)
	high = st.number_input("Enter your daily high temperature (in F):  ", min_value=None, max_value=None)
	low = st.number_input("Enter your daily low temperature (in F): ", min_value=None, max_value=None)
	rain = st.number_input("Enter your precipitation (in inches): ", min_value=None, max_value=None)

lat = str(np.round(lat,decimals=1))
lon = str(np.round(lon,decimals=1))
high = str((high - 32) * 5/9)
low = str((low - 32) * 5/9)
rain = str((rain * 25.4) * 10)

new_input = spark.createDataFrame([(1, lat, lon, high, low, rain)],
								T.StructType(
									[T.StructField("id",T.IntegerType(),True),
									T.StructField("LAT",T.StringType(),True),
									T.StructField("LON",T.StringType(),True),
									T.StructField("maxT",T.StringType(),True),
									T.StructField("minT",T.StringType(),True),
									T.StructField("rain",T.StringType(),True)]))

new_input = new_input.withColumn("LAT", new_input["LAT"].cast("double"))
new_input = new_input.withColumn("LON", new_input["LON"].cast("double"))
new_input = new_input.withColumn("maxT", new_input["maxT"].cast("double"))
new_input = new_input.withColumn("minT", new_input["minT"].cast("double"))
new_input = new_input.withColumn("precip", new_input["rain"].cast("double"))


va = VectorAssembler(inputCols=features2, outputCol = "features")

mm = MinMaxScaler(inputCol="features",outputCol="features_scaled")

c1, c2 = st.columns((1, 1, ))


#if new_input.select("rain") != 0.00:
new_input = va.transform(new_input)
mm = mm.fit(new_input)
new_scale = mm.transform(new_input)
new_pred = model.transform(new_scale)
prob = new_pred.select("probability").collect()
predi = new_pred.select("Prediction").collect()
proba = [row[0] for row in prob]
probs = []
for i in proba[0]:
	probs.append(i)
labels = ["Class A","Class B","Class C","Class D",
				"Class E","Class F", "Class G"]	
results = pd.DataFrame(list(zip(labels,probs)),columns=["Class of Fire","Probability"])
with c2:	
	st.markdown("# Classes of Fire")
	st.markdown(
		"**Class A** - one-fourth acre or less \n"
		)
	st.markdown(
		"**Class B** - more than one-fourth acre, but less than 10 acres \n"
		)
	st.markdown(
		"**Class C** - 10 acres or more, but less than 100 acres \n"
		)
	st.markdown(
		"**Class D** - 100 acres or more, but less than 300 acres \n"
		)
	st.markdown(
		"**Class E** - 300 acres or more, but less than 1,000 acres \n"
		)
	st.markdown(
		"**Class F** - 1,000 acres or more, but less than 5,000 acres \n"
		)
	st.markdown(
		"**Class G** - 5,000 acres or more \n"
		)
	
with c1:
	st.markdown("# Predictions")
	st.dataframe(results)

	st.markdown("*Based on data from 2000-2005 from [the US Department of Agriculture](https://www.fs.usda.gov/rds/archive/Catalog/RDS-2013-0009.5) and [the National Centers of Environmental Information](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00861/html)")

