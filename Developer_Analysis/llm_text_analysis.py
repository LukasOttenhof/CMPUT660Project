import pandas as pd
from datetime import datetime
import requests
import time

class TextAnalyzer:
    """
    Analyze developer text (GitHub or StackOverflow) for sentiment and topic trends over time using Gemini API.

    Attributes:
        api_key: Gemini API key
        text_type: 'github' or 'stackoverflow'
        file_path: path to CSV file containing developer text
        df: loaded data
    """

    GEMINI_ENDPOINT = "https://api.openai.com/v1/responses"  # Gemini API endpoint

    def __init__(self, api_key, text_type, file_path):
        self.api_key = api_key
        self.text_type = text_type
        self.file_path = file_path 
        self.df = pd.read_csv(file_path)

        if "text" not in self.df.columns:
            raise ValueError("CSV must contain a 'text' column with developer messages.")
        if "date" not in self.df.columns:
            raise ValueError("CSV must contain a 'date' column with timestamps.")

        self.df["date"] = pd.to_datetime(self.df["date"])

    # ------------------ PROMPTS ------------------
    sentiment_prompt_template = """
Classify the sentiment of this developer message as one of:
positive, neutral, or negative.

Examples:
"Added robust caching layer" → positive
"Fixed broken unit test" → neutral
"This workaround sucks" → negative

Text: "{text}"
Output only the label (positive, neutral, or negative).
"""

    topic_prompt_template = """
Classify the following developer message into one of these categories:
bug_fix, feature, refactor, documentation, performance, code implementation, other

Examples:
"Fix crash when loading config file" → bug_fix
"Add caching mechanism for models" → performance
"Update README with usage examples" → documentation

Text: "{text}"
Output only the label.
"""

    # ------------------ GEMINI API CALL ------------------
    def _query_gemini(self, prompt, max_retries=3):
        """
        Call Gemini API to get LLM completion.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gemini-1.5",  # replace with your preferred Gemini model
            "input": prompt
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(self.GEMINI_ENDPOINT, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                # Gemini response usually under 'output_text' or similar
                label = data.get("output_text") or data.get("choices", [{}])[0].get("text", "")
                return label.strip()
            except Exception as e:
                print(f"⚠️ Gemini API call failed ({attempt+1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
        return "unknown"

    # ------------------ ANALYSIS FUNCTIONS ------------------
    def analyze_sentiment(self, save_path=None):
        sentiments = []
        for text in self.df["text"]:
            prompt = self.sentiment_prompt_template.format(text=text)
            label = self._query_gemini(prompt)
            sentiments.append(label)
        self.df["sentiment"] = sentiments

        if save_path:
            self.df.to_csv(save_path, index=False)
            print(f"✅ Sentiment analysis saved to {save_path}")
        return self.df

    def analyze_topic(self, save_path=None):
        topics = []
        for text in self.df["text"]:
            prompt = self.topic_prompt_template.format(text=text)
            label = self._query_gemini(prompt)
            topics.append(label)
        self.df["topic"] = topics

        if save_path:
            self.df.to_csv(save_path, index=False)
            print(f"✅ Topic analysis saved to {save_path}")
        return self.df

    # ------------------ AGGREGATE OVER TIME ------------------
    def aggregate_over_time(self, freq="M"):
        agg_df = self.df.copy()
        agg_df["period"] = agg_df["date"].dt.to_period(freq)
        sentiment_counts = agg_df.groupby(["period", "sentiment"]).size().unstack(fill_value=0)
        topic_counts = agg_df.groupby(["period", "topic"]).size().unstack(fill_value=0)
        return sentiment_counts, topic_counts

if __name__ == "__main__":
    analyzer = TextAnalyzer(
    api_key="YOUR_GEMINI_API_KEY",
    text_type="github",
    file_path="developer_messages.csv"
    )

    # Sentiment
    analyzer.analyze_sentiment(save_path="sentiment_results.csv")

    # Topic
    analyzer.analyze_topic(save_path="topic_results.csv")

    # Trends over months
    sentiment_counts, topic_counts = analyzer.aggregate_over_time(freq="M")
    print(sentiment_counts)
    print(topic_counts)
