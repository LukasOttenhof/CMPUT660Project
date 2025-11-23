# from datetime import datetime, timezone
# import pandas as pd


# from numeric_repo_analysis import DeveloperAnalyzer as NumericAnalyzer
# from text_repo_analysis import DeveloperTextCollector as TextAnalyzer
# from datetime import datetime, timezone

# class UnifiedDeveloperAnalyzer:
#     def __init__(self, tokens, repo_name, start_date, end_date, max_workers=5):
#         print("Initializing unified developer analyzer...")
#         self.tokens = tokens if isinstance(tokens, list) else [tokens]
#         self.repo_name = repo_name
#         self.start_date = start_date
#         self.end_date = end_date
#         self.max_workers = max_workers
  

#         # Sub-analyzers

#         self.numeric_analyzer = NumericAnalyzer(
#             token=self.tokens[0],
#             repo_name=self.repo_name,
#             start_date=self.start_date,
#             end_date=self.end_date,
#             max_workers=max_workers
#         )
#         print("Numeric analyzer intitialized")

#         self.text_analyzer = TextAnalyzer(
#             tokens=self.tokens,
#             repo_name=self.repo_name,
#             start_date=self.start_date,
#             end_date=self.end_date,
#             max_workers=max_workers
#         )
#         print("Text analyzer analyzed")

  
#         print("Analysis class intitialized")

#     # ------------------ AUTHOR EXTRACTION ------------------
#     def get_authors(self, include_external=False, min_activities=5):
#         """
#         Collect main author names from numeric and text analyses.
#         Optionally filter to core collaborators or those with enough activity.
#         """
#         authors = set()

#         # --- From numeric analysis (multiple entries per author) ---
#         if hasattr(self.numeric_analyzer, "records") and self.numeric_analyzer.records:
#             numeric_df = pd.DataFrame(self.numeric_analyzer.records)
#             if not numeric_df.empty and "author" in numeric_df.columns:
#                 # Count activities per author
#                 counts = numeric_df["author"].value_counts()
#                 main_authors = counts[counts >= min_activities].index.tolist()
#                 authors.update(main_authors)

#         # --- From text analysis ---
#         for record in getattr(self.text_analyzer, "records", []):
#             if "author" in record:
#                 authors.add(record["author"])

#         # --- Optionally restrict to collaborators only ---
#         if not include_external:
#             try:
#                 core_collabs = [user.login for user in self.numeric_analyzer.repo.get_collaborators()]
#                 authors = {a for a in authors if a in core_collabs}
#             except Exception as e:
#                 print(f"⚠️ Could not fetch collaborators: {e}")

#         return sorted(authors)

#     # ------------------ MAIN EXECUTION ------------------
#     def run(self, save_path_prefix=None):
#         """
#         Run all analyses and combine results.
#         """
#         print(f"\n🚀 Running unified developer analysis for {self.repo_name}...\n")

#         # 1. Numeric metrics
#         print("Starting Numeric analysis")
#         numeric_df = self.numeric_analyzer.run(
#             save_path=f"{save_path_prefix}_numeric.csv" if save_path_prefix else None
#         )
#         print("Completed numeric analysis")

#         print("Retriving text data")
#         # 2. Textual data
#         text_df = self.text_analyzer.run(
#             save_path=f"{save_path_prefix}_text.csv" if save_path_prefix else None
#         )
#         print("Completed repo text data retrival")

#         print("Starting stack overflow data colelction")
#         # 3. StackOverflow data
#             # authors = self.get_authors()
#             # stack_df = self.fetch_stackoverflow_data(authors, save_path_prefix)

#         print("\n Repository developer analysis complete!")
#         return numeric_df, text_df


# if __name__ == "__main__":
#     # --- CONFIGURATION ---
#     print("Starting developer analysis pipeline")
#     GITHUB_TOKENS = [
#         "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",
#         "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",
#         "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"
#     ]

#     START_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
#     END_DATE = datetime(2025, 11, 1, tzinfo=timezone.utc)

#     SAVE_PREFIX = "Developer_Analysis/data/"
    
#     # --- Read repo list from TXT file ---
#     repo_file = r"/Users/lukas./Desktop/CMPUT660Project/repos_min1000pr.txt"
#     with open(repo_file, "r", encoding="utf-8") as f:
#         # Extract the full_name (first column before '|') from each line
#         repo_list = [line.split("|")[0].strip() for line in f.readlines()]
#     print("got repo list")
#     # --- RUN PIPELINE FOR EACH REPO ---

#     for repo_name in repo_list:
        
#         print("Initializing NumericAnalyzer...")
#         analyzer = UnifiedDeveloperAnalyzer(
#             tokens=GITHUB_TOKENS[0],
#             repo_name=repo_name, 
#             start_date=START_DATE,
#             end_date=END_DATE,
#             max_workers=1,
#         )
#         print("NumericAnalyzer initialized successfully")
   
#         numeric_df, text_df, stack_df = analyzer.run(save_path_prefix=SAVE_PREFIX)

#         print(f"\nSummary of results for {repo_name}:")
#         print(f"  Numeric records: {len(numeric_df)}")
#         print(f"  Text records: {len(text_df)}")
#         print(f"  StackOverflow records: {len(stack_df)}")

#         print(f"Files saved for {repo_name}:")
#         print(f"  - {SAVE_PREFIX}_numeric.csv")
#         print(f"  - {SAVE_PREFIX}_text.csv")
#         print(f"  - {SAVE_PREFIX}_stackoverflow_<author>.csv (individual per author)")
#         break

