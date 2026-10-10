import os
import pandas as pd
import streamlit as st
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder

from data_pipeline import load_kaggle_groceries, load_uploaded_csv, build_daily_demand, train_forecaster, make_forecast
from agents import run_agents, build_context, chat_with_ai, inspect_image, transcribe_audio, speak_text, create_inventory

st.set_page_config(page_title="Morrow | Zero-Waste Grocery AI", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#17251f; --muted:#68756e; --line:#e5eae6; --green:#dff0e4; --green2:#b9ddc4; --accent:#244b39; --cream:#fbfcfa; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
.stApp { background:#ffffff; }
[data-testid="stSidebar"] { background:#fbfcfa; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--ink); }
.block-container { padding-top:1.8rem; padding-bottom:3rem; max-width:1450px; }
h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; color:var(--ink); }
h1 { font-size:2.35rem !important; font-weight:800 !important; }
h2 { font-size:1.5rem !important; font-weight:750 !important; }
h3 { font-size:1.05rem !important; font-weight:700 !important; }
[data-testid="stMetric"] { background:#fbfcfa; border:1px solid var(--line); padding:18px 18px; border-radius:16px; }
[data-testid="stMetricLabel"] { color:#68756e; font-weight:600; }
[data-testid="stMetricValue"] { color:#17251f; font-family:'Manrope',sans-serif; }
.stButton > button, .stDownloadButton > button { background:#203d30; color:#fff; border:1px solid #203d30; border-radius:11px; padding:.58rem 1rem; font-weight:650; }
.stButton > button:hover, .stDownloadButton > button:hover { background:#315944; color:#fff; border-color:#315944; }
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div { border-radius:10px !important; }
[data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:15px; background:#fbfcfa; }
[data-testid="stTabs"] button { font-weight:650; }
hr { border-color:var(--line); }
.eyebrow { color:#4f795f; text-transform:uppercase; font-size:.72rem; letter-spacing:.14em; font-weight:800; }
.hero { background:linear-gradient(120deg,#f7fbf7,#edf7ef); border:1px solid #e0ece2; border-radius:22px; padding:24px 26px; margin-bottom:20px; }
.small-note { color:#68756e; font-size:.86rem; }
.pill { display:inline-block; background:#e4f2e7; color:#244b39; border-radius:999px; padding:5px 10px; font-size:.75rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

def get_api_key():
    try:
        return st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")

def get_openai_model():
    try:
        return st.secrets.get("OPENAI_MODEL", "") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    except Exception:
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

api_key = get_api_key().strip()
openai_model = get_openai_model().strip() or "gpt-4o-mini"
client = OpenAI(api_key=api_key) if api_key else None

if "chat" not in st.session_state:
    st.session_state.chat = []
if "inventory" not in st.session_state:
    st.session_state.inventory = None
if "source_label" not in st.session_state:
    st.session_state.source_label = "Not loaded"
if "forecast_model" not in st.session_state:
    st.session_state.forecast_model = None
if "daily_demand" not in st.session_state:
    st.session_state.daily_demand = None
if "agent_results" not in st.session_state:
    st.session_state.agent_results = None

with st.sidebar:
    st.markdown("## 🌿 Morrow")
    st.caption("Zero-waste grocery intelligence")
    st.divider()
    st.markdown("**Data source**")
    data_choice = st.radio("Choose your dataset", ["Public Kaggle grocery transactions", "Upload a Kaggle CSV"], label_visibility="collapsed")
    uploaded = None
    if data_choice == "Upload a Kaggle CSV":
        uploaded = st.file_uploader("Upload dataset (.csv)", type=["csv"])
    st.caption("Default source: public Groceries Dataset from Kaggle via KaggleHub.")
    load_btn = st.button("Load data & train forecast", use_container_width=True)
    st.divider()
    st.markdown("**AI connection**")
    if client:
        st.success(f"OpenAI connected · {openai_model}")
    else:
        st.warning("OpenAI API key not set")
        st.caption("Set OPENAI_API_KEY in .streamlit/secrets.toml or as an environment variable to enable AI chat, image analysis and voice.")
    st.caption(f"Chat and image model: {openai_model}")
    st.divider()
    st.markdown("**Demo assumptions**")
    st.caption("Sales transactions are real dataset records. Current stock, batch IDs, costs and expiry dates are demo simulations unless supplied by your own CSV.")

if load_btn:
    with st.spinner("Loading records and training the demand model..."):
        try:
            if data_choice == "Upload a Kaggle CSV":
                if uploaded is None:
                    st.error("Upload a CSV first.")
                    st.stop()
                raw, label = load_uploaded_csv(uploaded)
            else:
                raw, label = load_kaggle_groceries()
            daily, mapping = build_daily_demand(raw)
            model, metrics = train_forecaster(daily)
            forecast = make_forecast(model, daily, mapping)
            inventory = create_inventory(mapping, forecast)
            st.session_state.inventory = inventory
            st.session_state.daily_demand = daily
            st.session_state.forecast_model = model
            st.session_state.source_label = label
            st.session_state.model_metrics = metrics
            st.session_state.agent_results = run_agents(inventory)
            st.success(f"Loaded {len(raw):,} source rows. Forecast model trained on {len(daily):,} product-day records.")
        except Exception as e:
            st.error(f"Could not load/train dataset: {e}")
            st.info("Try the public Kaggle source again or upload a CSV containing a date column and a product/item column.")

inventory = st.session_state.inventory

st.markdown('<div class="eyebrow">Perishable operations · intelligence workspace</div>', unsafe_allow_html=True)
st.title("Waste less. Protect every margin.")
st.markdown("One AI workspace for demand, shelf-life risk, markdowns and store-to-store transfers.")

if inventory is None:
    st.markdown("""
    <div class="hero">
      <h3 style="margin-top:0">Start with real grocery transaction data</h3>
      <p style="margin-bottom:8px">Load the public Kaggle grocery transactions dataset or upload a larger Kaggle CSV. Morrow aggregates transactions by product and date, trains a demand forecasting model, then creates clearly labelled demo batches for operational recommendations.</p>
      <span class="pill">Real transaction history</span>&nbsp; <span class="pill">Model trained locally</span>&nbsp; <span class="pill">Simulated batch records disclosed</span>
    </div>
    """, unsafe_allow_html=True)
    a,b,c = st.columns(3)
    a.metric("Specialist agents", "4")
    b.metric("Forecasting", "Train on dataset")
    c.metric("AI assistant", "LLM + vision + voice")
    st.info("Use **Load data & train forecast** in the sidebar. The first download may take a little time.")
    st.stop()

daily = st.session_state.daily_demand
agent_results = st.session_state.agent_results or []
metrics = st.session_state.get("model_metrics", {})
total_stock = int(inventory["stock_units"].sum())
at_risk = int(inventory.loc[inventory["risk_level"] == "High", "stock_units"].sum())
stock_value = float((inventory["stock_units"] * inventory["unit_cost"]).sum())
product_count = int(inventory["product"].nunique())

m1,m2,m3,m4 = st.columns(4)
m1.metric("Products tracked", f"{product_count:,}")
m2.metric("Demo stock units", f"{total_stock:,}")
m3.metric("High-risk units", f"{at_risk:,}")
m4.metric("Estimated stock cost", f"₹{stock_value:,.0f}")
st.caption(f"Dataset: {st.session_state.source_label} · Forecast method: {metrics.get('model','HistGradientBoostingRegressor')} · Evaluation MAE: {metrics.get('mae','n/a')} units/day")
st.divider()

tab_overview, tab_agents, tab_assistant, tab_vision, tab_data = st.tabs(["Overview", "Agent decisions", "AI assistant", "Shelf image", "Dataset & model"])

with tab_overview:
    left, right = st.columns([1.3,1])
    with left:
        st.subheader("Demand outlook")
        plot_df = inventory[["product","forecast_daily_units"]].sort_values("forecast_daily_units", ascending=False).head(12).set_index("product")
        st.bar_chart(plot_df)
    with right:
        st.subheader("Shelf-life risk")
        risk_counts = inventory["risk_level"].value_counts().reindex(["High","Medium","Low"]).fillna(0).astype(int)
        st.bar_chart(risk_counts)
        st.caption("Risk is calculated against demo expiry dates. Replace these with real batch records before operational use.")
    st.subheader("Inventory explorer")
    search = st.text_input("Search product", placeholder="e.g. whole milk, yogurt, vegetables")
    view = inventory.copy()
    if search:
        view = view[view["product"].str.contains(search, case=False, na=False)]
    show_cols = ["product","forecast_daily_units","stock_units","days_to_expiry","risk_level","unit_cost","recommended_action"]
    st.dataframe(view[show_cols].sort_values(["risk_level","days_to_expiry"]), use_container_width=True, hide_index=True)
    st.download_button("Download inventory CSV", view.to_csv(index=False).encode("utf-8"), "morrow_inventory_demo.csv", "text/csv")

with tab_agents:
    st.subheader("Four specialist agents")
    st.caption("Agent calculations are grounded in the loaded data and demo batch assumptions. The LLM can explain results but cannot fabricate missing inventory facts.")
    for item in agent_results:
        with st.container(border=True):
            st.markdown(f"### {item['agent']}")
            st.markdown(f"**Finding:** {item['finding']}")
            st.write(item["recommendation"])
            st.caption(item["evidence"])
    if st.button("Refresh agent recommendations"):
        st.session_state.agent_results = run_agents(inventory)
        st.rerun()

with tab_assistant:
    st.subheader("Talk to Morrow")
    st.caption("Ask about demand, expiry risk, pricing or transfers. OpenAI API access is required for real LLM responses.")
    if not client:
        st.warning("AI chat is disabled until an API key is configured. The rest of the dashboard works without it.")
    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Ask about today's inventory decisions...")
    if prompt:
        st.session_state.chat.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            if client:
                context = build_context(inventory, agent_results, metrics)
                try:
                    answer = chat_with_ai(client, st.session_state.chat, context, model=openai_model)
                except Exception as e:
                    answer = f"AI request failed: {e}"
            else:
                answer = "I’m not connected to an LLM yet. Add your OPENAI_API_KEY in Streamlit secrets to enable natural-language reasoning."
            st.markdown(answer)
        st.session_state.chat.append({"role":"assistant","content":answer})
    if client and st.session_state.chat:
        last_assistant = next((m["content"] for m in reversed(st.session_state.chat) if m["role"]=="assistant"), "")
        if last_assistant:
            if st.button("🔊 Speak latest answer"):
                try:
                    audio = speak_text(client, last_assistant)
                    st.audio(audio, format="audio/mp3")
                except Exception as e:
                    st.error(f"Speech generation failed: {e}")
    if client:
        st.markdown("#### Speak to Morrow")
        st.caption("Allow microphone access in your browser, then record your question.")
        recorded_audio = audio_recorder(text="🎙️ Start recording", recording_color="#a23c3c", neutral_color="#203d30", icon_size="2x")
        if recorded_audio and st.button("Send recording to Morrow"):
            try:
                transcript = transcribe_audio(client, recorded_audio, "morrow_voice.wav")
                st.markdown(f"**You said:** {transcript}")
                st.session_state.chat.append({"role":"user","content":transcript})
                answer = chat_with_ai(client, st.session_state.chat, build_context(inventory, agent_results, metrics), model=openai_model)
                st.session_state.chat.append({"role":"assistant","content":answer})
                st.markdown(answer)
                st.audio(speak_text(client, answer), format="audio/mp3")
            except Exception as e:
                st.error(f"Voice workflow failed: {e}")
        voice_file = st.file_uploader("Or upload a voice recording (WAV/MP3/M4A)", type=["wav","mp3","m4a"], key="voice_upload")
        if voice_file and st.button("Transcribe uploaded recording"):
            try:
                transcript = transcribe_audio(client, voice_file.getvalue(), voice_file.name)
                st.markdown(f"**You said:** {transcript}")
                st.session_state.chat.append({"role":"user","content":transcript})
                answer = chat_with_ai(client, st.session_state.chat, build_context(inventory, agent_results, metrics), model=openai_model)
                st.session_state.chat.append({"role":"assistant","content":answer})
                st.markdown(answer)
                st.audio(speak_text(client, answer), format="audio/mp3")
            except Exception as e:
                st.error(f"Voice workflow failed: {e}")

with tab_vision:
    st.subheader("Inspect a shelf or product")
    st.write("Upload a shelf image, label or produce photo. Vision analysis can identify visible products and visible warning signs; it cannot prove internal freshness or replace an expiry label.")
    image_file = st.file_uploader("Choose image", type=["png","jpg","jpeg","webp"], key="shelf_image")
    question = st.text_area("What should the AI inspect?", value="Identify visible products and packaging. Note any visible damage, discoloration, wilting, leaks, or labels/expiry dates that can be read. Separate observations from uncertainty.", height=100)
    if image_file:
        st.image(image_file, use_container_width=True)
        if st.button("Analyze image with AI"):
            if not client:
                st.error("Add OPENAI_API_KEY to enable vision analysis.")
            else:
                with st.spinner("Analyzing image..."):
                    try:
                        result = inspect_image(client, image_file.getvalue(), image_file.type, question, inventory, model=openai_model)
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"Image analysis failed: {e}")

with tab_data:
    st.subheader("Dataset and training")
    st.write(f"**Source:** {st.session_state.source_label}")
    st.write(f"**Raw rows:** {len(daily):,} aggregated product-day records used after preprocessing")
    st.write(f"**Training model:** {metrics.get('model','n/a')}")
    st.write(f"**Mean absolute error:** {metrics.get('mae','n/a')}")
    st.write(f"**Products:** {inventory['product'].nunique():,}")
    st.markdown("### Data integrity")
    st.markdown("""
    - **Real:** product/date transaction occurrences from the loaded dataset.
    - **Trained:** a scikit-learn demand model fitted to the available product-day history.
    - **Simulated:** stock on hand, unit costs, batch IDs and expiration dates when the source dataset does not contain those fields.
    - **Not production-ready yet:** connect real store stock, supplier lead times, product shelf-life rules and actual batch records before making live pricing or transfer decisions.
    """)
    st.dataframe(daily.head(1000), use_container_width=True, hide_index=True)
