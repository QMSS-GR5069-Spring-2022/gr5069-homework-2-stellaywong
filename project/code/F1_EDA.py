# Databricks notebook

# import libraries and functions
from pyspark.sql.functions import datediff, current_date, avg, max, min, first, col, row_number
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window

# load datasets
df_pit_stops = spark.read.csv("s3://columbia-gr5069-main/raw/pit_stops.csv", header = True)
df_results = spark.read.csv("s3://columbia-gr5069-main/raw/results.csv", header = True)
df_drivers = spark.read.csv("s3://columbia-gr5069-main/raw/drivers.csv", header = True)
df_races = spark.read.csv("s3://columbia-gr5069-main/raw/races.csv", header = True)
df_driver_standings = spark.read.csv("s3://columbia-gr5069-main/raw/driver_standings.csv", header = True)

# Exploratory analysis
# 1. The average time each driver spent at the pit stop for each race
avg_pit_stops = df_pit_stops \
    .groupby("driverId", "raceId") \
    .agg(avg("duration"))

# 2. Rank the average time spent at the pit stop in order of who won each race
df_winners_pit_stops = df_results \
    .select("positionOrder", "driverId") \
    .join(avg_pit_stops, on=["driverId"]) \
    .sort(df_results.positionOrder.asc())

# 3. Insert the missing code (e.g: ALO for Alonso) for drivers based on the 'drivers' dataset
df_add_driver_codes = df_drivers \
    .select("code", "driverId") \
    .join(df_winners_pit_stops, on=["driverId"])
display(df_add_driver_codes)

# 4. The youngest and oldest driver for each race (with a new column called “Age”)
# find the age (difference the current day to their birthday)
df_drivers_age = df_drivers \
    .withColumn("age", datediff(current_date(), "dob"))

# every driver in every race
df_just_races = df_results \
    .select("raceId", "driverId")

# join to their race results
df_drivers_age = df_drivers_age \
    .select("age", "driverId", "forename", "surname")

df_drivers_just_ages = df_drivers_age \
    .join(df_just_races, on=["driverId"])

df_drivers_just_ages = df_drivers_just_ages \
    .groupby("raceId") \
    .agg(max("age"), min("age"))

# join back on names of oldest drivers, Fernando Alonso (F1 from Season 1 Episode 2) and Kimi Räikkönen
df_drivers_names_max_ages = df_drivers_just_ages \
    .join(df_drivers_age, df_drivers_just_ages['max(age)'] == df_drivers_age["age"])

# max(age) matches the driver's age
df_drivers_max = df_drivers_names_max_ages \
    .withColumn("oldest driverId", (df_drivers_names_max_ages["driverId"])) \
    .withColumn("oldest forename", (df_drivers_names_max_ages["forename"])) \
    .withColumn("oldest surname", (df_drivers_names_max_ages["surname"])) \
    .withColumn("max(age)", (df_drivers_names_max_ages["max(age)"])) \
    .drop("driverId", "forename", "surname", "min(age)")

# join back on names of youngest driver Nico Rosberg
df_drivers_names_min_ages = df_drivers_just_ages \
    .join(df_drivers_age, df_drivers_just_ages['min(age)'] == df_drivers_age["age"])

# min(age) matches the driver's age
df_drivers_min = df_drivers_names_min_ages \
    .withColumn("youngest driverId", (df_drivers_names_max_ages["driverId"])) \
    .withColumn("youngest forename", (df_drivers_names_max_ages["forename"])) \
    .withColumn("youngest surname", (df_drivers_names_max_ages["surname"])) \
    .withColumn("min(age)", (df_drivers_names_max_ages["min(age)"])) \
    .drop("driverId", "forename", "surname", "max(age)")

# convert their age to years -- we only do this now because earlier, two drivers could be the same "age" but younger/older within their calendar year
df_range_ages = df_drivers_max \
    .join(df_drivers_min, on=["raceId"]) \
    .withColumn("max(age)", (df_drivers_max["max(age)"]/365).cast(IntegerType())) \
    .withColumn("min(age)", (df_drivers_min["min(age)"]/365).cast(IntegerType())) \
    .drop("age", "age")

# 5. The driver with the most wins and losses for a race
# each driver ID comes with a name
df_drivers_names = df_drivers \
    .select("driverId", "forename", "surname")

# each driver ID has a win record per race. this record may change as the driver has more races.
df_driver_standings_wins = df_driver_standings \
    .select("driverId", "raceId", "wins")

# for each race, we see all drivers and their current number of wins
df_driver_standings_names = df_drivers_names \
    .join(df_driver_standings_wins, on=["driverId"])

# from this source of how to use windows https://sparkbyexamples.com/pyspark/pyspark-find-maximum-row-per-group-in-dataframe/
windowWins = Window.partitionBy("raceId").orderBy(col("wins").desc())
top = df_driver_standings_names \
    .withColumn("top_driver", row_number().over(windowWins)) \
    .filter(col("top_driver") == 1) \
    .drop("top_driver")

windowWorsts = Window.partitionBy("raceId").orderBy(col("wins").asc())
worst = df_driver_standings_names \
    .withColumn("worst_driver", row_number().over(windowWorsts)) \
    .filter(col("worst_driver") == 1) \
    .drop("worst_driver")

# rename the columns for clarity
renamed_top = top \
    .withColumn("top driverId", (top["driverId"])) \
    .withColumn("top forename", (top["forename"])) \
    .withColumn("top surname", (top["surname"])) \
    .withColumn("top wins", (top["wins"])) \
    .drop("driverId", "forename", "surname", "wins")

renamed_worst = worst \
    .withColumn("worst driverId", (worst["driverId"])) \
    .withColumn("worst forename", (worst["forename"])) \
    .withColumn("worst surname", (worst["surname"])) \
    .withColumn("fewest wins", (worst["wins"])) \
    .drop("driverId", "forename", "surname", "wins")

most_fewest_wins = renamed_top \
    .join(renamed_worst, on=["raceId"])

# 6. Raise and answer a question.
# Does the average Formula 1 driver get younger each year?

# find the age (difference the current day to their birthday)
df_drivers_age = df_drivers \
    .withColumn("age", datediff(current_date(), "dob"))

# convert their age to years
df_drivers_age = df_drivers_age \
    .withColumn("age", (df_drivers_age["age"] / 365.25).cast(IntegerType()))

# join to their races
df_drivers_age = df_drivers_age \
    .select("age", "driverId", "forename", "surname") \
    .join(df_add_driver_codes, on=["driverId"])

# find the max and min age per race
df_drivers_just_ages = df_drivers_age \
    .groupby("raceId") \
    .agg(max("age"), min("age"))

df_races_years = df_races \
    .select("raceId", "year", "name") \
    .join(df_drivers_just_ages, on=["raceId"])

df_races_years = df_races_years \
    .withColumn("range", df_races_years["max(age)"] - df_races_years["min(age)"])

# It seems the maximum and minimum age have both dropped a decade over the last 10 years, but the range has remained about the same! Potential reasons include, new technology that allows the team to instruct drivers when to pit (so less racing experience is required), or the longer F1 season (from 7 races in 1950 to 21 races today) which may favor competitors who are willing to undergo the travel and have less to lose to racing full-time.
