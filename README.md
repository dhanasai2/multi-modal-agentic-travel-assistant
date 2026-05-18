# Multi-Modal Agentic Travel Assistant

LangGraph + Streamlit travel assistant that routes between:
1. **Local vector store** (Paris, Tokyo, New York facts)
2. **Web search path** (mock web search for unseen cities)

It returns a **structured JSON object** and renders:
- city summary
- weather forecast line chart
- image gallery

Due to API-access limitations, **Groq-hosted Llama models are used as a drop-in replacement for OpenAI/Claude APIs**.  
The architecture remains fully compatible with OpenAI-style tool-calling and LangGraph orchestration workflows.

## Tech stack

- LangGraph (typed state, conditional routing, checkpointer memory)
- Streamlit (GUI)
- Groq Llama (OpenAI-style tool calls)
- FAISS local vector store
- Mock APIs for weather/search/images with simulated latency

## Graph highlights

- **Conditional switch**: vector path vs web path vs weather-only follow-up
- **Manual tool execution**: parses `tool_calls`, runs function, appends `ToolMessage` manually
- **Parallel fan-out**: weather and image fetch nodes run concurrently
- **Memory/time travel**: checkpointer keeps context by `thread_id`; follow-up weather questions reuse city summary

## Project structure

```text
.
|-- app.py
|-- requirements.txt
|-- .env.example
|-- graph.png                  # generated via script
|-- graph.mmd                  # generated via script
|-- scripts/
|   `-- generate_graph_png.py
`-- src/
    `-- travel_agent/
        |-- __init__.py
        |-- graph.py
        |-- mock_apis.py
        |-- models.py
        `-- vector_store.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add your Groq key to `.env`:

```env
GROQ_API_KEY=...
```

## Run

```bash
streamlit run app.py
```

## Generate graph visualization

```bash
python scripts\generate_graph_png.py
```

This creates:
- `graph.mmd`
- `graph.png`

## Expected demo flow

1. Ask: `Tell me about Tokyo`  
   - routes to vector store summary tool
   - fetches weather + images in parallel

2. Ask: `What about next week?`  
   - keeps city context (`Tokyo`) from memory
   - updates weather without regenerating summary

3. Ask: `Tell me about Kyoto`  
   - routes to mock web search tool (not in vector store)
