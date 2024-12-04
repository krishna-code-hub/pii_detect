import os
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, lit, explode
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, FloatType, BooleanType
from presidio_analyzer import AnalyzerEngine
import pandas as pd
import yaml

# Set Python paths for PySpark
os.environ['PYSPARK_PYTHON'] = r"C:\Users\Krishna\anaconda3\envs\piidetect\python.exe"
os.environ['PYSPARK_DRIVER_PYTHON'] = r"C:\Users\Krishna\anaconda3\envs\piidetect\python.exe"
os.environ['SPARK_LOCAL_DIRS'] = r"C:\temp\spark"

# Load PII configuration from external config file (e.g., config.yaml)
with open('pii_config.yaml', 'r') as f:
    pii_config = yaml.safe_load(f)

# Start Spark session with improved configuration to prevent deadlocks
spark = SparkSession.builder \
    .appName("PIIDetection") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .config("spark.network.timeout", "600s") \
    .config("spark.executor.heartbeatInterval", "100s") \
    .getOrCreate()

# Define regex patterns for common PII types
regex_patterns = {
    "EMAIL": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$",
    "PHONE": r"\b\d{10}\b|\+\d{1,3}\s\d{10}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b"
}

# Define regex detection function
def regex_detect(text: str) -> list:
    pii_hits = []
    if isinstance(text, str):  # Check if the input is a string
        for pii_type, pattern in regex_patterns.items():
            if re.search(pattern, text):
                pii_hits.append((pii_type, 0.8, "regex"))  # Add method "regex" for detection method
    return pii_hits

# Define PII detection function that initializes Presidio within each task lazily
def detect_pii(text: str) -> list:
    if text is None:
        return []

    if not isinstance(text, str):
        text = str(text)

    # Lazy initialization for AnalyzerEngine to prevent repeated initialization overhead
    global analyzer
    if 'analyzer' not in globals():
        analyzer = AnalyzerEngine()

    # Apply regex detection
    regex_results = regex_detect(text)

    # Apply Presidio detection
    presidio_results = analyzer.analyze(text=text, entities=[], language="en")
    presidio_hits = [(result.entity_type, result.score, "presidio") for result in presidio_results]

    # Combine results
    all_results = regex_results + presidio_hits
    return all_results

# Define column name detection function
def column_name_detect(column_name: str) -> list:
    pii_hits = []
    for pii_name, patterns in pii_config['pii_columns'].items():
        if column_name.lower() in patterns:
            pii_hits.append((pii_name, 1.0, "column_name"))  # High confidence for column name match
            break  # Stop after the first match for each PII type
    return pii_hits

# Define Pandas UDF to detect PII in a series with smaller partition sizes
@pandas_udf(ArrayType(StructType([
    StructField("pii_type", StringType(), True),
    StructField("confidence", FloatType(), True),
    StructField("method", StringType(), True)  # Add method field to indicate detection approach
])))
def detect_pii_series(s: pd.Series) -> pd.Series:
    return s.apply(detect_pii)

# Load sample CSV file into DataFrame and repartition to reduce load per partition
input_df = spark.read.csv("data/sample.csv", header=True, inferSchema=True).repartition(10)

# Display input DataFrame to show sample records
input_df.show(20, truncate=False)

# Define the schema for column-level PII DataFrame
column_pii_schema = StructType([
    StructField("table_name", StringType(), True),
    StructField("column_name", StringType(), True),
    StructField("data_type", StringType(), True),
    StructField("pii_detected", BooleanType(), True),
    StructField("pii_type", StringType(), True),
    StructField("confidence", FloatType(), True),
    StructField("detection_method", StringType(), True)
])

# Apply PII detection on each column and collect results
results = []

# Table name for the output (assuming "sample_table" as a placeholder)
table_name = "sample_table"

for column in input_df.columns:
    # Detect based on column name
    column_pii_hits = column_name_detect(column)
    if column_pii_hits:  # Only create DataFrame if there are column-level PII hits
        column_pii_df = spark.createDataFrame([(table_name, column, input_df.schema[column].dataType.simpleString(), True, pii_hit[0], pii_hit[1], pii_hit[2]) for pii_hit in column_pii_hits], schema=column_pii_schema)
        results.append(column_pii_df)

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
        col("pii_info.confidence").alias("confidence"),
        col("pii_info.method").alias("detection_method")  # Include the detection method in the output
    )

    # Add results to the list
    results.append(pii_data)

# Union all the individual column results
if results:
    final_result_df = results[0]
    for result_df in results[1:]:
        final_result_df = final_result_df.union(result_df)
    # Show final results
    final_result_df.show(500, truncate=False)
else:
    print("No PII data detected.")
