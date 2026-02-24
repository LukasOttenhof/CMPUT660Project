import os
from data_loader import load_all

TARGETS = [
    'noperson83/WBee-appware', 
    'dBinet/SquirrelAttack', 
    'supermarsx/smtp-burst'
]

def audit_repos():
    data = load_all()

    for category in data:
        # Get the 'repo' column/series from the dataframe
        # We use .values to check the actual strings inside
        repo_column = data[category]['repo']
        
        # Check if any of our TARGETS are in this category's repo list
        for target in TARGETS:
            if target in repo_column.values:
                print(f"Found {target} in {category}")

if __name__ == "__main__":
    audit_repos()