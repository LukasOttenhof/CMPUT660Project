import requests
import time
import pandas as pd
from datetime import datetime
import sys

# --- CONFIG ---
API_KEY = "rl_FvkLZcXVovTGcbe86uHGt6adk"  # replace with your real key
FILTER_WITH_BODY = "!9_bDDxJY5"
SLEEP_TIME = 1


class StackExchangeUserCollector:
    def __init__(self, user_id=None, username=None, site="stackoverflow", page_size=100):
        if not user_id and not username:
            raise ValueError("You must provide either a user_id or username.")

        self.site = site
        self.page_size = page_size
        self.records = []

        if username:
            print(f"🔍 Resolving username '{username}'...")
            self.user_id = self._resolve_user_id_from_name(username)
            if not self.user_id:
                raise ValueError(f"No user found with name '{username}' on {site}.")
            print(f"✅ Resolved user '{username}' → ID {self.user_id}")
        else:
            self.user_id = user_id

        if not self._user_exists():
            raise ValueError(f"❌ User {self.user_id} does not exist on {site}.")

    def _resolve_user_id_from_name(self, username):
        url = "https://api.stackexchange.com/2.3/users"
        params = {"site": self.site, "inname": username, "key": API_KEY, "pagesize": 1}
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            print("⚠️ Failed to resolve username.")
            return None
        items = resp.json().get("items", [])
        return items[0]["user_id"] if items else None

    def _fetch(self, url, params):
        """Fetch data with retry/backoff handling and a visible spinner."""
        spinner = "|/-\\"
        i = 0
        max_pages = 100
        while True:
            sys.stdout.write(f"\rFetching {url.split('/')[-1]}... {spinner[i % len(spinner)]}")
            sys.stdout.flush()
            i += 1

            params["key"] = API_KEY
            resp = requests.get(url, params=params)
            data = resp.json()

            if "backoff" in data:
                wait = data["backoff"]
                print(f"\n⏳ API backoff requested. Sleeping for {wait}s...")
                time.sleep(wait)

            if resp.status_code == 200:
                sys.stdout.write("\r" + " " * 60 + "\r")  # clear line
                return data
            if i >= max_pages:
                break

            print(f"\n⚠️ Error {resp.status_code}: retrying in {SLEEP_TIME}s...")
            time.sleep(SLEEP_TIME)

    def _user_exists(self):
        url = f"https://api.stackexchange.com/2.3/users/{self.user_id}"
        params = {"site": self.site, "key": API_KEY}
        resp = requests.get(url, params=params)
        return len(resp.json().get("items", [])) > 0

    def _fetch_items(self, endpoint, item_type):
        page = 1
        while True:
            url = f"https://api.stackexchange.com/2.3/users/{self.user_id}/{endpoint}"
            params = {"site": self.site, "pagesize": self.page_size, "page": page, "filter": FILTER_WITH_BODY}
            data = self._fetch(url, params)
            items = data.get("items", [])

            print(f"📄 Page {page}: fetched {len(items)} {item_type}(s)")

            if not items:
                break

            for obj in items:
                self.records.append({
                    "type": item_type,
                    "title": obj.get("title", ""),
                    "body": obj.get("body", ""),
                    "date": datetime.utcfromtimestamp(obj["creation_date"])
                })

            if not data.get("has_more", False):
                print(f"✅ No more pages for {item_type}s.")
                break

            page += 1
            time.sleep(SLEEP_TIME)

    def run(self, save_path=None):
        save_path = "Developer_Analysis/data/" + save_path if save_path else None
        for endpoint, item_type in [("questions", "question"), ("answers", "answer"), ("comments", "comment")]:
            print(f"\n=== Fetching {item_type}s ===")
            self._fetch_items(endpoint, item_type)

        print(f"\n✅ Collected {len(self.records)} total records for user {self.user_id}")

        df = pd.DataFrame(self.records)
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"💾 Saved to {save_path}")
        return df


if __name__ == "__main__":
    collector = StackExchangeUserCollector(username="Jon Skeet", site="stackoverflow")
    df = collector.run(save_path="stackexchange_user_posts.csv")
    print(df.head())
