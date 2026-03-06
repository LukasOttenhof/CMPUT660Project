# An Empirical Study of Activity Change in Agent-Assisted Repositories

A guide to the supplementary materials used in this paper.

---

## Repository Groups

| Group | Directory | Description |
|-------|-----------|-------------|
| **Agentic** | `inputs/agentic/` | Repositories that adopted agent-assisted PRs as defined in the AIDev dataset. Data is split at the first agentic PR date. |
| **Human Baseline** | `inputs/human_2/` | Repositories with human-authored PRs used as a comparison baseline. |

Within each group, data is split into before and after periods to allow for three-way analysis (before, after agentic, after developer-only).

---

## Produced Datasets

Both `inputs/agentic/` and `inputs/human_2/` contain the following parquet files (both before and after variations for each):

| File | Contents |
|------|----------|
| `commits_before/after.parquet` | repo, author, date, SHA, lines added/deleted, files changed |
| `commit_messages_before/after.parquet` | commit message text, repo, author, date, SHA |
| `pull_requests_before/after.parquet` | author, date, activity type (created/merged), PR number, time to merge (hours) |
| `pr_bodies_before/after.parquet` | pr body, repo, author, date, PR number |
| `reviews_before/after.parquet` | review text, author, date, PR number, review state |
| `review_comments_before/after.parquet` | review comment text, PR number and review state |
| `issues_before/after.parquet` | author, date, activity type (opened/closed), issue number, time to close (hours) |
| `issue_bodies_before/after.parquet` | issue text, author, date, issue number |
| `repo_month_complexity_detailed.parquet` | per-repo per-month cyclomatic complexity metrics |

Discussion parquets were also made but data for them was never populated due to issues obtaining it through the GitHub API. 

Additional files in `inputs/agentic/`:
- `complexity_summary.csv` — Complexity statistics per repo
- `cache_analysis_report.csv` — Cache hit/miss report from complexity extraction

Manually produced topic analysis files (project root):
- `manual_commit_message_topic_mapping.csv` — Maps BERTopic clusters of commit messages to meta-topics
- `manual_pr_body_topic_mapping.csv` — Maps BERTopic clusters of pull request bodies to meta-topics

---

## Pipeline: Run Order

### Phase 1 — Cloning/Extraction 
This process is not necessary to rerun for data analysis now that parquets have already been generated once as it is quite long.

- **`scripts/build_dataset/clone_repos.py`**
- **`scripts/build_dataset/repo_analysis.py`**

The repo_analysis script is currently set to run for the human baseline, however, the file path can be changed to filtered_repos_3year50pr.txt to re-generate our dataset from the agentic repository subset we used.

---

### Phase 2 — Dataset Summary (Optional)

**`scripts/build_dataset/data_summary.py`**

Generates an overview of the extracted dataset.

---

### Phase 3 — Analysis

Analysis scripts use `scripts/analysis/data_loader.py` as a shared foundation, which loads all parquet files from both groups and returns a unified dictionary keyed by data type and period (e.g., `commit_messages_before`, `commit_messages_after_human`, `commit_messages_after_agent`).

The following sections can be run in any order (no strict ordering between RQs):

#### RQ1 — Activity Volume

| Script | Description |
|--------|-------------|
| `scripts/analysis/analyze_rq1.py` | Summary tables and boxplots for commits, PRs, issues, and text length before vs. after |
| `scripts/analysis/rq1_lin_reg.py` | Linear regression over activity metrics |

#### RQ2 — Developer Activity Shifts

| Script | Description |
|--------|-------------|
| `scripts/analysis/rq2_wilcox.py` | Wilcoxon signed-rank tests and chi-square tests for activity-per-developer-per-month |
| `scripts/analysis/rq2_box_plot.py` | Box plot visualizations for RQ2 merge time metrics |
| `scripts/analysis/rq2_web_plot.py` | Creates pokemon-style web plot |

#### RQ3 — Bug/Fix Categorization

| Script | Description |
|--------|-------------|
| `scripts/analysis/bug_analysis.py` | Categorizes commit messages and PR bodies containing bug/fix/refactor keywords. Runs chi-square tests with Cramér's V effect size to compare category distributions across periods. |

#### RQ4 — Code Complexity

| Script | Description |
|--------|-------------|
| `scripts/analysis/build_code_complexity.py` | Extracts code complexity metrics (via Lizard) by checking out each commit month by month for human repositories.
| `scripts/analysis/build_code_complexity.ipynb` | Extracts code complexity metrics (via Lizard) by checking out each commit month by month for agentic repositories.
| `scripts/analysis/code_complexity_analyze.ipynb` | Analyzes extracted code complexity.

#### RQ5 — Topic Distribution

| Script | Description |
|--------|-------------|
| `scripts/analysis/analyze_rq5_topic_BERT.py` | BERTopic run over commit messages and PR bodies (3-way: before / after-human / after-agent) must be run to generate clusters before manually mapping to meta-topics |
| `scripts/analysis/rq5_topics_full_analysis.py` | Full analysis using manually produced mappings; creates LaTeX tables and chi-square results used in paper |

#### RQ6 — Sentiment Analysis

| Script | Description |
|--------|-------------|
| `scripts/analysis/analyze_rq6_vadersentiment.py` | VADER sentiment scoring for commits, PR bodies, issues, and discussions |
| `scripts/analysis/rq6_plot_stats.py` | Visualizations of sentiment distributions |
| `scripts/analysis/rq6_table_stats.py` | Statistical summary tables and LaTeX tables of sentiment distributions |

---

## Utility Scripts

| Script | Description |
|--------|-------------|
| `scripts/analysis/inspect_parquets.py` | Outlines parquet structure, column types, missing values, and text length distributions for debugging |
| `scripts/analysis/table_utils.py` | Shared helpers for computing statistics and exporting to CSV/LaTeX tables |
| `scripts/analysis/plot_utils.py` | Shared helpers for plotting |