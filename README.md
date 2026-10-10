# Morrow — Zero-Waste Grocery AI

A Streamlit prototype for perishable demand forecasting and waste reduction. It includes a real OpenAI-powered chat, image analysis, audio transcription and speech generation, a scikit-learn demand forecasting model trained on loaded transaction data, and four inventory agents.

## Important data distinction

- **Real dataset:** the public Kaggle `heeraldedhia/groceries-dataset` transaction file, fetched through KaggleHub at runtime, or a CSV uploaded by you.
- **Model training:** product/day transaction counts are derived from historical transactions and used to fit a scikit-learn `HistGradientBoostingRegressor`.
- **Simulated demo fields:** stock on hand, unit costs, batch IDs and expiry dates are generated placeholders because the default transaction dataset does not provide real batch-level inventory or expiry records.
- **Not yet a production system:** connect real POS, batch inventory, store locations, supplier lead times, retail prices, and verified expiry dates before using recommendations operationally.

## 1. Run locally in VS Code (Windows PowerShell)

Open this folder in VS Code and open a terminal in this folder.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, use the venv Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Click **Load data & train forecast** in the sidebar. The first run downloads the public Kaggle dataset. KaggleHub requires internet access and Kaggle's terms apply.

## 2. Enable real AI, vision and voice

Create `.streamlit/secrets.toml` locally:

```toml
OPENAI_API_KEY = "paste-your-own-api-key-here"
OPENAI_MODEL = "gpt-4o-mini"
```

The app uses the OpenAI API for real chat and image analysis when the key is configured. `gpt-4o-mini` is the default vision-capable chat model; change `OPENAI_MODEL` to another model available to your account. You can also configure these values as environment variables. Do not commit this file or share the key. API usage may cost money; it is separate from free hosting.

## 3. Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload these files.
2. Ensure `.streamlit/secrets.toml` is not uploaded.
3. Open https://share.streamlit.io/ and choose **Create app**.
4. Select your repository, branch, and `app.py`.
5. In **Advanced settings → Secrets**, add:

```toml
OPENAI_API_KEY = "your-key"
OPENAI_MODEL = "gpt-4o-mini"
```

6. Deploy. The first launch may take several minutes because Python packages and the dataset are downloaded.

## Kaggle dataset options

Default, manageable starting point:
- `heeraldedhia/groceries-dataset` — grocery purchase records with date and item descriptions. Suitable for transaction-frequency demand proxies, not verified units sold or expiry history.

For larger forecasting experiments:
- [Kaggle Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) — about 18.7 MB of CSV files, with date, store, item and sales history. Suitable for a larger supervised forecasting experiment, but not specifically perishable grocery.
- [Kaggle Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data) — large real grocery-chain data with daily unit sales, store/item metadata, promotion and perishable-item information. It is roughly 480 MB compressed and may require accepting competition rules. Free-hosting memory/time limits can be exceeded. For this dataset, create a smaller product/store/date aggregate in Kaggle or locally, then upload the reduced CSV.

The upload reader currently caps imported CSVs at 1,000,000 rows to reduce memory risk. It recognizes common date/product/quantity fields including `date`, `item`, `family`, `sales`, and `unit_sales` where available.

The app currently uses the manageable grocery transactions dataset by default and supports uploading another Kaggle CSV with date/product columns (and optional quantity). For a dataset with store/item/sales fields, map the correct columns if their names differ.

## Architecture

- `data_pipeline.py`: dataset ingestion, cleaning, aggregation, model training, forecast generation.
- `agents.py`: four specialist decision agents plus LLM chat, vision and audio API helpers.
- `app.py`: Streamlit user interface and app orchestration.
