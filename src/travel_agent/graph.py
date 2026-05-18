import os
import re
from difflib import get_close_matches
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from .mock_apis import get_location_images, get_weather_forecast, mock_web_search
from .models import TravelResponse, WeatherPoint
from .vector_store import city_exists, retrieve_city_facts, vector_city_list

load_dotenv()


class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    city: str
    route: Literal["vector", "web", "weather_only"]
    weather_days: int
    city_summary: str
    knowledge_source: Literal["vector_store", "web_search", "memory"]
    weather_forecast: list[dict]
    image_urls: list[str]
    final_response: dict
    error: str


@tool
def vector_city_lookup(city: str) -> str:
    """Lookup city facts from the local vector store."""
    facts = retrieve_city_facts(city)
    return facts or f"No local facts found for {city}."


@tool
def web_city_search(city: str) -> str:
    """Lookup city facts from a web-like search source."""
    return mock_web_search(city)


def _last_user_message(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _normalize_city(candidate: str, known_pool: set[str]) -> str | None:
    raw = re.sub(r"\s+", " ", candidate.strip().lower())
    raw_no_space = raw.replace(" ", "")
    aliases = {
        "toyko": "tokyo",
        "tokio": "tokyo",
        "nyc": "new york",
        "newyork": "new york",
    }
    if raw_no_space in aliases:
        return aliases[raw_no_space].title()
    if raw in known_pool:
        return raw.title()
    close = get_close_matches(raw, list(known_pool), n=1, cutoff=0.8)
    if close:
        return close[0].title()
    return None


def _extract_city(query: str, previous_city: str | None = None) -> tuple[str | None, bool]:
    q = query.strip()
    q_lower = q.lower()
    known_pool = set(vector_city_list()) | {"kyoto", "snohomish", "barcelona"}

    for city in sorted(known_pool, key=len, reverse=True):
        if city in q_lower:
            return city.title(), True

    match = re.search(r"(?:about|in|for|to)\s+([A-Za-z\s]+)", q, flags=re.IGNORECASE)
    if match:
        candidate = re.sub(r"[^A-Za-z\s]", "", match.group(1)).strip()
        candidate = re.sub(
            r"\b(next week|this week|weather|forecast|please|tell me)\b",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        if candidate:
            normalized = _normalize_city(candidate, known_pool)
            if normalized:
                return normalized, True
            return candidate.title(), True

    if previous_city:
        return previous_city, False
    return None, False


def _is_weather_followup(query: str, has_new_city: bool) -> bool:
    if has_new_city:
        return False
    q = query.lower()
    weather_signals = ["next week", "weather", "forecast", "temperature", "rain", "humid"]
    return any(signal in q for signal in weather_signals)


def _init_llm(model_name: str, temperature: float = 0.2) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")
    return ChatGroq(model=model_name, temperature=temperature, api_key=api_key)


def prepare_context(state: TravelState) -> TravelState:
    query = _last_user_message(state.get("messages", []))
    city, has_new_city = _extract_city(query=query, previous_city=state.get("city"))
    if not city:
        raise ValueError("No city could be inferred from the request.")

    weather_only = _is_weather_followup(query, has_new_city)
    weather_days = 7 if "next week" in query.lower() else 5

    if weather_only and state.get("city_summary"):
        route = "weather_only"
    elif city_exists(city):
        route = "vector"
    else:
        route = "web"

    return {"user_query": query, "city": city, "route": route, "weather_days": weather_days}


def _run_manual_tool_summary(
    *,
    llm: ChatGroq,
    city: str,
    lookup_tool,
    source: Literal["vector_store", "web_search"],
) -> TravelState:
    planner = llm.bind_tools([lookup_tool], tool_choice="any")
    user_prompt = (
        f"City: {city}. Call the available tool once, then produce a concise city summary "
        "for travelers in exactly 4 sentences."
    )
    ai_plan = planner.invoke(
        [
            SystemMessage(content="You must call the tool before writing the summary."),
            HumanMessage(content=user_prompt),
        ]
    )

    tool_messages: list[ToolMessage] = []
    if ai_plan.tool_calls:
        for call in ai_plan.tool_calls:
            if call["name"] != lookup_tool.name:
                continue
            payload = call.get("args", {})
            result = lookup_tool.invoke(payload)
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"], name=call["name"])
            )
        final_messages = [
            SystemMessage(
                content="Use only tool output. Keep it human, practical, and under 90 words."
            ),
            HumanMessage(content=user_prompt),
            ai_plan,
            *tool_messages,
        ]
    else:
        fallback = lookup_tool.invoke({"city": city})
        final_messages = [
            SystemMessage(
                content="Use only tool output. Keep it human, practical, and under 90 words."
            ),
            HumanMessage(content=user_prompt),
            AIMessage(content=f"Tool output: {fallback}"),
        ]
    final = llm.invoke(final_messages)

    return {
        "city_summary": str(final.content).strip(),
        "knowledge_source": source,
        "messages": [ai_plan, *tool_messages, AIMessage(content=str(final.content))],
    }


def summarize_from_vector(state: TravelState, llm: ChatGroq) -> TravelState:
    return _run_manual_tool_summary(
        llm=llm,
        city=state["city"],
        lookup_tool=vector_city_lookup,
        source="vector_store",
    )


def summarize_from_web(state: TravelState, llm: ChatGroq) -> TravelState:
    return _run_manual_tool_summary(
        llm=llm,
        city=state["city"],
        lookup_tool=web_city_search,
        source="web_search",
    )


def weather_fanout(_: TravelState) -> TravelState:
    return {}


async def fetch_weather(state: TravelState) -> TravelState:
    forecast = await get_weather_forecast(state["city"], days=state.get("weather_days", 5))
    return {"weather_forecast": forecast}


async def fetch_images(state: TravelState) -> TravelState:
    images = await get_location_images(state["city"], count=4)
    return {"image_urls": images}


def aggregate_results(state: TravelState) -> TravelState:
    return state


def finalize_response(state: TravelState) -> TravelState:
    source = state.get("knowledge_source", "memory")
    summary = state.get("city_summary", "")
    if state.get("route") == "weather_only":
        source = "memory"
    payload = TravelResponse(
        city=state["city"],
        city_summary=summary,
        weather_forecast=[WeatherPoint(**row) for row in state.get("weather_forecast", [])],
        image_urls=state.get("image_urls", []),
        source=source,
    )
    return {"final_response": payload.model_dump()}


def finalize_weather_only(state: TravelState) -> TravelState:
    return finalize_response(state)


def route_after_prepare(state: TravelState) -> str:
    return state["route"]


def build_workflow(model_name: str = "llama-3.3-70b-versatile") -> StateGraph:
    llm = _init_llm(model_name=model_name)
    graph = StateGraph(TravelState)

    graph.add_node("prepare_context", prepare_context)
    graph.add_node("summarize_from_vector", lambda s: summarize_from_vector(s, llm))
    graph.add_node("summarize_from_web", lambda s: summarize_from_web(s, llm))
    graph.add_node("weather_fanout", weather_fanout)
    graph.add_node("fetch_weather", fetch_weather)
    graph.add_node("fetch_weather_only", fetch_weather)
    graph.add_node("fetch_images", fetch_images)
    graph.add_node("aggregate_results", aggregate_results)
    graph.add_node("finalize_response", finalize_response)
    graph.add_node("finalize_weather_only", finalize_weather_only)

    graph.add_edge(START, "prepare_context")
    graph.add_conditional_edges(
        "prepare_context",
        route_after_prepare,
        {
            "vector": "summarize_from_vector",
            "web": "summarize_from_web",
            "weather_only": "fetch_weather_only",
        },
    )
    graph.add_edge("summarize_from_vector", "weather_fanout")
    graph.add_edge("summarize_from_web", "weather_fanout")
    graph.add_edge("weather_fanout", "fetch_weather")
    graph.add_edge("weather_fanout", "fetch_images")
    graph.add_edge("fetch_weather", "aggregate_results")
    graph.add_edge("fetch_images", "aggregate_results")
    graph.add_edge("aggregate_results", "finalize_response")
    graph.add_edge("fetch_weather_only", "finalize_weather_only")
    graph.add_edge("finalize_response", END)
    graph.add_edge("finalize_weather_only", END)
    return graph


def compile_graph(model_name: str = "llama-3.3-70b-versatile"):
    workflow = build_workflow(model_name=model_name)
    return workflow.compile(checkpointer=MemorySaver())
