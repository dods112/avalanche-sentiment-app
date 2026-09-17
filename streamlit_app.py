import os
import io
import datetime as dt

import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from openai import OpenAI
from fpdf import FPDF

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"

st.set_page_config(page_title="Avalanche Sentiment Insights", layout="wide", page_icon="❄️")
st.title("❄️ Avalanche Sentiment Insights")
st.caption("Fast prototyping of a GenAI-powered customer review analyzer")

# Groq's API is OpenAI-compatible, so we reuse the OpenAI client with a different base_url
client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1") if API_KEY else None

# ---------------------------------------------------------
# DATA LOADING & CLEANING
# ---------------------------------------------------------
@st.cache_data
def load_data(path="data/customer_reviews.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["SUMMARY"] = df["SUMMARY"].fillna("").astype(str)
    df["SENTIMENT_SCORE"] = pd.to_numeric(df["SENTIMENT_SCORE"], errors="coerce")
    df = df.dropna(subset=["DATE", "SENTIMENT_SCORE"])
    df["MONTH"] = df["DATE"].dt.to_period("M").astype(str)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Could not find data/customer_reviews.csv. Please add the dataset first.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("Filters")
products = sorted(df["PRODUCT"].dropna().unique())
selected_products = st.sidebar.multiselect("Product", products, default=products)

min_date, max_date = df["DATE"].min(), df["DATE"].max()
date_range = st.sidebar.date_input("Date range", (min_date, max_date))

sentiment_range = st.sidebar.slider("Sentiment score range", -1.0, 1.0, (-1.0, 1.0))

filtered = df[
    (df["PRODUCT"].isin(selected_products))
    & (df["SENTIMENT_SCORE"].between(sentiment_range[0], sentiment_range[1]))
]
if len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = filtered[(filtered["DATE"] >= start) & (filtered["DATE"] <= end)]

avg_sentiment = filtered["SENTIMENT_SCORE"].mean() if len(filtered) else 0
worst_product = (
    filtered.groupby("PRODUCT")["SENTIMENT_SCORE"].mean().idxmin()
    if len(filtered) else "N/A"
)

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab_overview, tab_deep_dive, tab_ai = st.tabs(["📊 Overview", "🔍 Deep Dive", "🤖 AI Insights"])

# ===========================================================
# TAB 1: OVERVIEW
# ===========================================================
with tab_overview:
    if len(filtered) == 0:
        st.info("No reviews match the current filters. Try widening the sidebar filters.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Sentiment", f"{avg_sentiment:.2f}")
        col2.metric("Most Negative Product", worst_product)
        col3.metric("Total Reviews", len(filtered))

        st.subheader("Average Sentiment by Product")
        bar_data = filtered.groupby("PRODUCT")["SENTIMENT_SCORE"].mean().reset_index()
        fig_bar = px.bar(bar_data, x="PRODUCT", y="SENTIMENT_SCORE", color="SENTIMENT_SCORE",
                          color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Sentiment Over Time")
        line_data = filtered.groupby("DATE")["SENTIMENT_SCORE"].mean().reset_index()
        fig_line = px.line(line_data, x="DATE", y="SENTIMENT_SCORE")
        st.plotly_chart(fig_line, use_container_width=True)

# ===========================================================
# TAB 2: DEEP DIVE (heatmap + raw data)
# ===========================================================
with tab_deep_dive:
    st.subheader("🔥 Sentiment Heatmap: Product vs Month")
    heatmap_data = (
        filtered.groupby(["PRODUCT", "MONTH"])["SENTIMENT_SCORE"]
        .mean()
        .reset_index()
        .pivot(index="PRODUCT", columns="MONTH", values="SENTIMENT_SCORE")
    )
    if heatmap_data.empty:
        st.info("Not enough data to build a heatmap for the current filters.")
    else:
        fig_heatmap = px.imshow(
            heatmap_data,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            labels=dict(x="Month", y="Product", color="Avg Sentiment"),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.subheader("Product Reviews")
    st.markdown("""
<style>
.review-table-wrap {
    border: 1px solid #2A3441;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.review-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.review-table th {
    text-align: left;
    padding: 12px 16px;
    color: #8A94A3;
    font-weight: 500;
    border-bottom: 1px solid #2A3441;
    background-color: #161C26;
}
.review-table td {
    padding: 12px 16px;
    border-bottom: 1px solid #1F2733;
    color: #E5E9EF;
}
.review-table tr:last-child td { border-bottom: none; }
.review-table tr:hover td { background-color: #1A222E; }
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-dot {
    width: 6px; height: 6px; border-radius: 50%;
}
.badge-positive { background-color: #1E3A2B; color: #4ADE80; }
.badge-positive .badge-dot { background-color: #4ADE80; }
.badge-neutral { background-color: #2E2A1A; color: #E8C468; }
.badge-neutral .badge-dot { background-color: #E8C468; }
.badge-negative { background-color: #3A1E1E; color: #F87171; }
.badge-negative .badge-dot { background-color: #F87171; }
</style>
""", unsafe_allow_html=True)

    def sentiment_badge(score):
        if score >= 0.3:
            label, cls = "Positive", "badge-positive"
        elif score <= -0.3:
            label, cls = "Negative", "badge-negative"
        else:
            label, cls = "Neutral", "badge-neutral"
        return f'<span class="badge {cls}"><span class="badge-dot"></span>{label}</span>'

    table_rows = ""
    for _, row in filtered.sort_values("DATE", ascending=False).head(100).iterrows():
        summary_preview = (row["SUMMARY"][:70] + "…") if len(row["SUMMARY"]) > 70 else row["SUMMARY"]
        table_rows += (
            f"<tr>"
            f"<td>{row['PRODUCT']}</td>"
            f"<td>{row['DATE'].strftime('%d %b %Y')}</td>"
            f"<td>{summary_preview}</td>"
            f"<td>{row['SENTIMENT_SCORE']:.2f}</td>"
            f"<td>{sentiment_badge(row['SENTIMENT_SCORE'])}</td>"
            f"</tr>"
        )

    table_html = (
        '<div class="review-table-wrap">'
        '<table class="review-table">'
        '<thead><tr>'
        '<th>Product</th><th>Date</th><th>Review</th><th>Score</th><th>Sentiment</th>'
        '</tr></thead>'
        f'<tbody>{table_rows}</tbody>'
        '</table>'
        '</div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)
    if len(filtered) > 100:
        st.caption(f"Showing 100 most recent of {len(filtered)} reviews.")

    with st.expander("View full raw data (all columns)"):
        st.dataframe(filtered, use_container_width=True)

# ===========================================================
# TAB 3: AI INSIGHTS (chatbot + exec summary + pdf)
# ===========================================================
with tab_ai:
    st.subheader("💬 Ask About the Data")
    st.caption('Try: "Which product has the most negative reviews?" or "What\'s the trend over time?"')

    def ask_genai(question, context_df):
        if client is None:
            return "⚠️ No API key found. Please set GROQ_API_KEY in your .env file."
        sample = context_df.sample(min(30, len(context_df))) if len(context_df) else context_df
        context_text = sample[["PRODUCT", "DATE", "SUMMARY", "SENTIMENT_SCORE"]].to_string(index=False)
        prompt = f"""You are a data analyst assistant. Use this sample of customer review data to answer the question.
Data sample:
{context_text}

Question: {question}
Answer concisely and reference specific products or trends where relevant."""
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            answer = response.choices[0].message.content
            if not answer or not answer.strip():
                return "⚠️ The model returned an empty response. Please try asking again."
            return answer
        except Exception as e:
            return f"⚠️ API error: {e}"

    # Chat history lives in session_state so it persists across reruns
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render every past message
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input pinned at the bottom, like a real chat app
    user_question = st.chat_input("Type a question about the reviews...")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.spinner("Thinking..."):
            answer = ask_genai(user_question, filtered)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.subheader("📋 Executive Summary")

    def generate_executive_summary(context_df):
        if client is None:
            return "⚠️ No API key found. Please set GROQ_API_KEY in your .env file."
        negative_reviews = context_df.sort_values("SENTIMENT_SCORE").head(50)
        sample_text = negative_reviews[["PRODUCT", "SUMMARY", "SENTIMENT_SCORE"]].to_string(index=False)
        prompt = f"""You are a data analyst. Based on these most negative customer reviews, identify the TOP 3 complaints.
For each complaint, give: a short title, which product(s) it affects, and a 1-sentence description.
Format as a numbered list. Keep it under 150 words total.

Reviews:
{sample_text}"""
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            answer = response.choices[0].message.content
            if not answer or not answer.strip():
                return "⚠️ The model returned an empty response. Please try again."
            return answer
        except Exception as e:
            return f"⚠️ API error: {e}"

    if "exec_summary" not in st.session_state:
        st.session_state.exec_summary = None

    if st.button("Generate Executive Summary"):
        with st.spinner("Analyzing top complaints..."):
            st.session_state.exec_summary = generate_executive_summary(filtered)

    if st.session_state.exec_summary:
        st.markdown(st.session_state.exec_summary)

    st.divider()
    st.subheader("📄 Download PDF Report")

    def build_pdf_report(avg_sentiment, worst_product, total_reviews, summary_text):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Avalanche Sentiment Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Key Metrics", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, f"Average Sentiment: {avg_sentiment:.2f}", ln=True)
        pdf.cell(0, 7, f"Most Negative Product: {worst_product}", ln=True)
        pdf.cell(0, 7, f"Total Reviews: {total_reviews}", ln=True)
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Executive Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        text = summary_text if summary_text else "No summary generated yet. Click 'Generate Executive Summary' above first."
        # encode safely to latin-1 for fpdf2's default fonts
        safe_text = text.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe_text)

        return bytes(pdf.output(dest="S"))

    pdf_bytes = build_pdf_report(avg_sentiment, worst_product, len(filtered), st.session_state.exec_summary)
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="avalanche_sentiment_report.pdf",
        mime="application/pdf",
    )