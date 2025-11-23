import csv
from collections import defaultdict

file_path = "Developer_Analysis/data/results_top5_authors_only.csv"

# Using defaultdict is cleaner than manually checking if the key exists
author_counts = defaultdict(int)

# Use the 'csv' module for reliable parsing, especially with quoted commas
with open(file_path, "r", encoding="utf-8", newline="") as f:
    # Use csv.reader to handle CSV formatting correctly
    reader = csv.reader(f)

    # Skip the header row
    next(reader)

    # Iterate through the data rows
    for row in reader:
        # The first item in the row list (index 0) is the name
        if row: # Ensure the row is not empty
            author_name = row[0].strip()
            author_counts[author_name] += 1

# Convert to a regular dict and sort by count (optional, but helpful)
sorted_counts = dict(sorted(author_counts.items(), key=lambda item: item[1], reverse=True))

print("--- Unique Names and Their Frequencies ---")
for author, count in sorted_counts.items():
    print(f"{author}: {count}")