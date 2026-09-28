import pandas as pd

from agents.preprocessing_agent import preprocessing_agent
from tools.preprocessing import apply_preprocessing


df = pd.read_csv("data/cleaned_dataset.csv")

# Step 1: Agent decides the preprocessing plan
plan = preprocessing_agent(df)

print("\n")
print("PLAN:")
print(plan)

# Step 2: Executor applies the plan
preprocessed_df = apply_preprocessing(df, plan)

print("\n")
print("PREPROCESSED DATA:")
print(preprocessed_df.head())

print("\nShape:")
print(preprocessed_df.shape)