# AIML-Cohort-8
SDG 8 Decent Work and Economic Growth
Realtion bw  internet penetration and GDP per capita

import pandas as pd
import numpy as np

df = pd.read_csv("8f35d745-2ed9-4ca4-b7ea-f355e6fd00c5_Data.csv")

df.replace("..", np.nan, inplace=True)

id_cols = ["Country Name", "Series Name"]
year_cols = [col for col in df.columns if "YR" in col]

df = df[id_cols + year_cols]

df_long = df.melt(
    id_vars=["Country Name", "Series Name"],
    var_name="Year",
    value_name="Value"
)

df_long["Year"] = df_long["Year"].str.extract(r"(\d{4})")
df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")

df_clean = df_long.pivot_table(
    index=["Country Name", "Year"],
    columns="Series Name",
    values="Value"
).reset_index()

df_clean.columns.name = None

df_clean.rename(columns={
    "GDP per capita (current US$)": "GDP_per_Capita_USD",
    "Individuals using the Internet (% of population)": "Internet_Usage_%"
}, inplace=True)

df_clean.dropna(
    how="all",
    subset=["GDP_per_Capita_USD", "Internet_Usage_%"],
    inplace=True
)

gdp_wide = df_clean.pivot(
    index="Country Name",
    columns="Year",
    values="GDP_per_Capita_USD"
)

internet_wide = df_clean.pivot(
    index="Country Name",
    columns="Year",
    values="Internet_Usage_%"
)

gdp_wide.columns = [f"GDP_{col}" for col in gdp_wide.columns]
internet_wide.columns = [f"Internet_{col}" for col in internet_wide.columns]

df_wide = pd.concat([gdp_wide, internet_wide], axis=1).reset_index()

df_wide.to_csv("Cleaned_Wide_Format_Data.csv", index=False)
