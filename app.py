import os
import sys
import uuid

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from travel_agent import compile_graph  # noqa: E402


st.set_page_config(page_title="Multi-Modal Travel Assistant", page_icon=":earth_africa:", layout="wide")


@st.cache_resource
def load_agent(model_name: str):
    return compile_graph(model_name=model_name)


def render_weather_chart(forecast: list[dict]) -> None:
    if not forecast:
        st.info("No forecast data available.")
        return
    frame = pd.DataFrame(forecast)
    frame = frame.rename(columns={"date": "Date", "temp_c": "Temperature (degC)"})
    frame["Temperature (degC)"] = pd.to_numeric(frame["Temperature (degC)"], errors="coerce")
    frame = frame.dropna(subset=["Temperature (degC)"])
    if frame.empty:
        st.warning("Weather data format was invalid for chart rendering.")
        return
    chart_frame = frame.set_index("Date")[["Temperature (degC)"]]
    st.line_chart(chart_frame, height=280, use_container_width=True)
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_gallery(image_urls: list[str]) -> None:
    if not image_urls:
        st.info("No images found.")
        return
    cols = st.columns(min(4, len(image_urls)))
    for idx, url in enumerate(image_urls):
        with cols[idx % len(cols)]:
            try:
                st.image(url, use_container_width=True)
            except Exception:
                st.caption(f"Image unavailable: {url}")


def main() -> None:
    st.title("Multi-Modal Travel Assistant")
    st.caption("LangGraph + Streamlit + Groq Llama (OpenAI-style tool-calling)")

    with st.sidebar:
        st.subheader("Settings")
        model_name = st.selectbox(
            "Groq Model",
            options=[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ],
            index=0,
        )
        st.markdown(
            "This build uses Groq-hosted Llama models as a drop-in replacement for "
            "OpenAI/Claude APIs while preserving OpenAI-style tool-calling patterns."
        )

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    graph = load_agent(model_name=model_name)

    prompt = st.chat_input("Try: Tell me about Kyoto")
    if not prompt:
        st.info("Ask for a city to get a summary, weather trend, and image gallery.")
        return

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and gathering travel data..."):
            try:
                result = graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )
            except ValueError as exc:
                st.error(str(exc))
                return
        payload = result.get("final_response")
        if not payload:
            st.error("The assistant could not produce a structured response.")
            return

        st.subheader(payload["city"])
        st.write(payload["city_summary"])
        st.caption(f"Source path: {payload['source']}")

        left, right = st.columns([1.25, 1])
        with left:
            st.subheader("Weather Outlook")
            render_weather_chart(payload.get("weather_forecast", []))
        with right:
            st.subheader("City Gallery")
            render_gallery(payload.get("image_urls", []))


if __name__ == "__main__":
    main()
