import pandas as pd

# Use r'' for a raw string so Windows backslashes are handled correctly
file_path = r"G:\CMPUT660Project\inputs\50prs\commits_before.parquet"

# Load the file
df = pd.read_parquet(file_path)

# Print the first 5 rows
print(len(df))