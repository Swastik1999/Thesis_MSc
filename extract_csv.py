import pandas as pd

# The list of IDs you want to extract
target_ids = [
    1,5,11,14,20,27,34,38,40,41,44,53,54,58,66,70,76,78,79,88,90,93,94,95,100,
    105,108,114,117,120,123,124,125,129,132,133,142,144,145,146,148,151,153,
    157,158,161,164,165,166,168,173,177,179,182,186,188,190,191,208,209,210,
    223,228,231,233,235,240,241,242,243,252,258,259,261,263,265,269,277,279,
    284,286,292,297,298,300,302,307,308,316,317,318,319,321,322,328,330,332,
    333,337,338,341,345,359,362,363,365,367,368,369,371,374,379,384,391,398,
    404,406,409,415,423,425,431,436,439,440,447,454,455,456,458,460,467,468,
    470,474,478,481,482,485,486,491,492,493,499
]

# 1. Load the original CSV
df = pd.read_csv('Data/rag_test.csv')

# 2. Filter the DataFrame to only include rows where the 'id' is in our list
filtered_df = df[df['id'].isin(target_ids)]

# 3. Save the filtered rows to a new CSV file without the index column
filtered_df.to_csv('Data/extracted_file.csv', index=False)

print(f"Successfully extracted {len(filtered_df)} rows.")