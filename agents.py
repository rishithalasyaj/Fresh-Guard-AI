import base64
import pandas as pd
import numpy as np
from datetime import date

PERISHABLE_HINTS = ("milk","yogurt","curd","cheese","cream","meat","fish","chicken","lettuce","spinach","tomato","banana","apple","bread","berries","vegetable","fruit","egg")

def create_inventory(products, forecast):
    """Build clearly labelled demo stock/batch data when no real inventory feed exists."""
    rows = []
    today = pd.Timestamp.today().normalize()
    forecast_map = dict(zip(forecast["product"], forecast["forecast_daily_units"]))
    for i, product in enumerate(products):
        p = str(product)
        f = max(float(forecast_map.get(p, 0)), 0)
        # Stable deterministic demo assumptions, not actual stock measurements.
        stock = max(3, int(round(f * (3 + (i % 5)))))
        days = [1,2,3,4,5,7,9,12,14][i % 9]
        perishable = any(h in p.lower() for h in PERISHABLE_HINTS)
        if not perishable:
            days = max(days, 10)
        cost = round(18 + (i % 13) * 7.25, 2)
        risk = "High" if days <= 2 else ("Medium" if days <= 5 else "Low")
        if days <= 2:
            action = "Markdown review"
        elif f > 0 and stock / max(f, .25) > 7:
            action = "Transfer review"
        elif f > 0 and stock / max(f, .25) < 1.5:
            action = "Replenishment review"
        else:
            action = "Monitor"
        rows.append({
            "product":p, "forecast_daily_units":round(f,2), "stock_units":stock,
            "days_to_expiry":days, "expiry_date":(today + pd.Timedelta(days=days)).date().isoformat(),
            "batch_id":f"DEMO-{i+1:04d}", "unit_cost":cost, "risk_level":risk,
            "recommended_action":action, "record_type":"SIMULATED batch/stock"
        })
    return pd.DataFrame(rows)

def run_agents(inventory):
    if inventory is None or inventory.empty:
        return []
    high = inventory[inventory["risk_level"] == "High"]
    medium = inventory[inventory["risk_level"] == "Medium"]
    overstock = inventory[(inventory["forecast_daily_units"] > 0) & (inventory["stock_units"] / inventory["forecast_daily_units"].clip(lower=.25) > 7)]
    low_cover = inventory[(inventory["forecast_daily_units"] > 0) & (inventory["stock_units"] / inventory["forecast_daily_units"].clip(lower=.25) < 1.5)]
    markdown_products = high.sort_values("days_to_expiry")["product"].head(5).tolist()
    transfer_products = overstock["product"].head(5).tolist()
    replenish_products = low_cover["product"].head(5).tolist()
    return [
        {"agent":"01 · Demand Forecasting Agent","finding":f"{len(low_cover)} product groups show less than 1.5 days of stock cover.", "recommendation":"Prioritize replenishment review for: " + (", ".join(replenish_products) if replenish_products else "No clear low-cover products from this dataset."), "evidence":"Forecasts are trained on historical product-day sales/transaction counts. Dataset quantity semantics depend on the source."},
        {"agent":"02 · Spoilage & Shelf-Life Agent","finding":f"{len(high)} high-risk and {len(medium)} medium-risk demo batches.", "recommendation":"Inspect batches with 2 days or less remaining first. High-risk examples: " + (", ".join(markdown_products) if markdown_products else "none"), "evidence":"Expiry dates and batch IDs are simulated placeholders unless actual batch records are uploaded."},
        {"agent":"03 · Markdown Optimization Agent","finding":f"{len(high)} products need markdown review based on simulated expiry proximity.", "recommendation":"Review targeted discounts for high-risk batches; choose markdown depth using real margin, demand elasticity and local pricing before execution.", "evidence":"This prototype does not change retail prices. Margin optimization requires actual retail prices and price-response data."},
        {"agent":"04 · Inter-Store Transfer Agent","finding":f"{len(overstock)} product groups have more than 7 days of stock cover.", "recommendation":"Check nearby stores' demand and expiry dates before transferring: " + (", ".join(transfer_products) if transfer_products else "No clear overstock candidates."), "evidence":"Transfers are recommendations only; destination-store stock and travel time are not connected yet."},
    ]

def build_context(inventory, agents, metrics):
    summary = inventory[["product","forecast_daily_units","stock_units","days_to_expiry","risk_level","unit_cost","recommended_action"]].to_dict("records")
    return f"""You are Morrow, a careful zero-waste grocery operations assistant.
Use only facts in the context below. Clearly say when stock, batch, expiry, costs or forecast information is simulated.
Never claim an action was executed. Recommendations require human approval.
Explain uncertainty. Do not invent store locations, weather impacts, sales prices, margin, expiry dates, or supplier lead times.
Forecast model metadata: {metrics}
Inventory rows: {summary[:250]}
Agent findings: {agents}
"""

def chat_with_ai(client, messages, context, model="gpt-4o-mini"):
    api_messages = [{"role":"system","content":(
        "You are Morrow, an expert grocery inventory and food-waste operations assistant. "
        "Give concise, actionable answers. Explain which specialist agent should act and why. "
        "Do not invent missing values or state that a recommendation was executed. "
        "Batch-level stock/expiry fields may be simulated. Context follows:\n" + context
    )}]
    for msg in messages[-12:]:
        if msg["role"] in ("user","assistant"):
            api_messages.append({"role":msg["role"],"content":msg["content"]})
    response = client.chat.completions.create(model=model, messages=api_messages, temperature=0.2)
    return response.choices[0].message.content or "I couldn't generate a response."

def inspect_image(client, image_bytes, mime_type, question, inventory, model="gpt-4o-mini"):
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    context_products = ", ".join(inventory["product"].head(40).astype(str).tolist())
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You inspect grocery shelf images. Separate visible evidence from guesses. Do not claim internal freshness or exact shelf life from appearance alone. If labels are illegible, say so. Never infer inventory database values from an image."},
            {"role":"user","content":[
                {"type":"text","text":question + "\nKnown product groups in dataset (not necessarily visible in image): " + context_products},
                {"type":"image_url","image_url":{"url":f"data:{mime_type};base64,{encoded}","detail":"high"}}
            ]}
        ],
        max_tokens=700,
        temperature=0.2
    )
    return response.choices[0].message.content or "No analysis returned."

def transcribe_audio(client, audio_bytes, filename):
    result = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=(filename, audio_bytes)
    )
    return result.text

def speak_text(client, text):
    response = client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=text[:4000], response_format="mp3")
    return response.read()
