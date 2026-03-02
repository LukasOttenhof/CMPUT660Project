import glob

path = r'G:\CMPUT660Project\inputs\human_2\incremental\*_prs_before.parquet'
file_count = len(glob.glob(path))

print(f"Total files matching pattern: {file_count}")