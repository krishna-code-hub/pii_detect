import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, lit, explode, collect_list
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, FloatType, BooleanType
from presidio_analyzer import AnalyzerEngine
import pandas as pd
import os


os.environ['PYSPARK_PYTHON'] = r"C:\Users\Krishna\anaconda3\envs\piidetect\python.exe"
os.environ['PYSPARK_DRIVER_PYTHON'] = r"C:\Users\Krishna\anaconda3\envs\piidetect\python.exe"
os.environ['SPARK_LOCAL_DIRS'] = r"C:\temp\spark"


# Start Spark session with Spark NLP
spark = SparkSession.builder \
    .appName("PIIDetection") \
    .getOrCreate()

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

# Initialize Presidio Analyzer
analyzer = AnalyzerEngine()

# Broadcast Presidio to worker nodes
broadcasted_analyzer = spark.sparkContext.broadcast(analyzer)

# Define regex patterns for common PII types
regex_patterns = {
    "EMAIL": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$",
    "PHONE": r"\b\d{10}\b|\+\d{1,3}\s\d{10}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b"
}


# Define regex detection function
def regex_detect(text: str) -> list:
    pii_hits = []
    for pii_type, pattern in regex_patterns.items():
        if re.search(pattern, text):
            pii_hits.append((pii_type, 0.8))  # Confidence for regex can be set based on pattern strength
    return pii_hits


# Define Presidio-based detection function
def presidio_detect(text: str) -> list:
    analyzer = broadcasted_analyzer.value
    results = analyzer.analyze(text=text, entities=[], language="en")
    return [(result.entity_type, result.score) for result in results]


# Combine NER and PII Detection Function
def detect_pii(text: str) -> list:
    if text is None:
        return []

    # Apply regex detection
    regex_results = regex_detect(text)

    # Apply Presidio detection
    presidio_results = presidio_detect(text)

    # Combine results
    all_results = regex_results + presidio_results
    return all_results


# Define Pandas UDF to detect PII in a series
@pandas_udf(ArrayType(StructType([
    StructField("pii_type", StringType(), True),
    StructField("confidence", FloatType(), True)
])))
def detect_pii_series(s: pd.Series) -> pd.Series:
    return s.apply(detect_pii)


# Load sample CSV file into DataFrame
input_df = spark.read.csv("data/sample.csv", header=True, inferSchema=True)

# Apply PII detection on each column and collect results
results = []

# Table name for the output (assuming "sample_table" as a placeholder)
table_name = "sample_table"

for column in input_df.columns:
    # Add a new column with PII detection results
    pii_hits_df = input_df.withColumn(f"{column}_pii_hits", detect_pii_series(col(column)))

    # Extract results for each column
    pii_data = pii_hits_df.select(
        lit(table_name).alias("table_name"),
        lit(column).alias("column_name"),
        lit(input_df.schema[column].dataType.simpleString()).alias("data_type"),
        explode(col(f"{column}_pii_hits")).alias("pii_info")
    ).select(
        "table_name",
        "column_name",
        "data_type",
        (col("pii_info.pii_type").isNotNull()).alias("pii_detected"),
        col("pii_info.pii_type").alias("pii_type"),
        col("pii_info.confidence").alias("confidence")
    )

    # Add results to the list
    results.append(pii_data)

# Union all the individual column results
final_result_df = results[0]
for result_df in results[1:]:
    final_result_df = final_result_df.union(result_df)

# Show final results
final_result_df.show(truncate=False)
