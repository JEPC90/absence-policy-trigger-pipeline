# Enterprise Absence Policy Trigger Pipeline

An end-to-end automated data engineering pipeline built using **Medallion Architecture (Bronze -> Silver -> Gold)** to detect and report employee absence policy breaches in corporate environments.

## 🚀 Key Features & Technical Highlights
- **Medallion Data Architecture:** Clear separation of concerns with a raw ingestion tier (Bronze), a rigorous cleaning and validation tier (Silver), and an analytical aggregation tier (Gold).
- **Quarantine Gateway:** Automated schema validation that traps corrupted records, invalid data types, and orphaned attendance entries into isolated audit logs without breaking the pipeline.
- **Business-Day Gaps & Islands Algorithm:** Uses NumPy `busday_count` to map continuous absence spells while intelligently bridging weekends.
- **Master Orchestration:** Single-command execution (`python main.py`) with comprehensive logging.

## 🛠️ Tech Stack
- **Language:** Python
- **Core Libraries:** Pandas, NumPy
- **Logging & Error Handling:** Python `logging`, `try-except` execution wrappers

## ⚙️ Quickstart Guide
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/absence-policy-trigger-pipeline.git](https://github.com/YourUsername/absence-policy-trigger-pipeline.git)
   cd absence-policy-trigger-pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main pipeline:
   ```bash
   python main.py
   ```
