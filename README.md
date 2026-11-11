# ✈️ TripPilot: Travel Multi-Agent App

An AI travel planner built with **LangGraph**. You describe a trip in plain English, for example *"Flights from Mumbai to Dubai and hotels near Dubai Marina"*. A team of agents then finds live flights, looks up hotels, drafts an itinerary and writes a final travel plan, all shown in a **Streamlit** web app.

## How it works

```mermaid
flowchart LR
    U([User query]) --> F[✈️ flight_agent]
    F --> H[🏨 hotel_agent]
    H --> I[🗺️ itinerary_agent]
    I --> FA[✨ final_agent]
    FA --> R([Final travel response])

    F -. get_flights .-> AV[(Aviationstack API)]
    H -. tavily_search .-> TV[(Tavily Search API)]
    F & I & FA -. LLM .-> G[(Groq · gpt-oss-120b)]
    FA -. checkpoints .-> PG[(PostgreSQL)]
```

| Agent | What it does | Writes to state |
|---|---|---|
| `flight_agent` | Uses the LLM to pick out departure/arrival airports (city → IATA code), then fetches up to 5 live flights | `flight_results` |
| `hotel_agent` | Searches the web for hotels matching the request (top 5 results) | `hotel_results` |
| `itinerary_agent` | Drafts a day-by-day plan from the flight and hotel data | `itinerary` |
| `final_agent` | Combines everything into a Markdown reply: summary, flights table, hotels, itinerary and tips | `final_response` |

All agents share a `TravelState` (see [`main.py`](main.py)). Each step is saved to **PostgreSQL** through LangGraph's `PostgresSaver`, keyed by a `thread_id` (one per traveller).

## Features

- **Live progress:** the UI ticks off each agent as it finishes.
- **Flight cards:** route, airport names, times, terminal/gate, delay, and a colour-coded status badge, with a status filter.
- **Hotel cards:** short description, relevance score and a link to each result.
- **Itinerary and summary tabs**, with the plan downloadable as Markdown.
- **Per-traveller history:** conversations are saved to Postgres by traveller name.
- **Clear error handling:** missing airports or API errors show as plain-language tips, not stack traces.

## Project structure

```
.
├── main.py              # TravelState, agents (nodes), LangGraph graph, Postgres checkpointer
├── frontend.py          # Streamlit UI
├── tools/
│   ├── flight_tool.py   # Aviationstack: get_flights(dep_iata, arr_iata, limit=5)
│   └── tavily_tool.py   # Tavily: tavily_search(query, max_results=5)
├── requirements.txt
└── .env.example         # Template for your API keys
```

## Getting started

### Prerequisites

- Python 3.12+
- A running **PostgreSQL** database
- API keys for:
  - [Groq](https://console.groq.com/keys): the LLM (keys start with `gsk_`)
  - [Aviationstack](https://aviationstack.com/): flight data
  - [Tavily](https://tavily.com/): web search for hotels

### 1. Clone and install

```bash
git clone https://github.com/saupawar/Travel_Agent_Multi_app.git
cd Travel_Agent_Multi_app

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the template and fill in your own values:

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

```env
GROQ_API_KEY=gsk_your_groq_key
AVIATIONSTACK_API_KEY=your_aviationstack_key
tavily_API_KEY=tvly-your_tavily_key
DATABASE_URL=postgresql://user:password@localhost:5432/travel
```

The checkpoint tables are created automatically the first time the app runs.

### 3. Run

**Web app (recommended):**

```bash
streamlit run frontend.py
```

Then open <http://localhost:8501>.

**Command line:**

```bash
python main.py
```

## Example prompts

- *Flights from Mumbai to Dubai and good hotels near Dubai Marina*
- *New York to London trip, show flights and central hotels*
- *Delhi to Singapore, family-friendly hotels and a 3-day plan*

Name both the departure and arrival cities (or airport codes) so the flight agent can find a route.

## Configuration notes

- **LLM model:** set in `main.py` (`openai/gpt-oss-120b`). Groq retires models from time to time; if you see a `model_not_found` error, pick another model from your [Groq console](https://console.groq.com/docs/models). `openai/gpt-oss-20b` is a faster option.
- **Aviationstack free plan:** supports **HTTP only** (not HTTPS) and has a small monthly request quota. Each trip uses one request.
- **Result limits:** both tools return at most 5 results (`MAX_RESULTS` in each tool file).

## Troubleshooting

| Problem | Fix |
|---|---|
| `localhost refused to connect` | The Streamlit server isn't running. Start it with `streamlit run frontend.py` and keep that terminal open. |
| `ModuleNotFoundError` | Activate the virtual environment, then `pip install -r requirements.txt`. |
| `Invalid API Key` (Groq) | Check `GROQ_API_KEY` in `.env`; Groq keys start with `gsk_`. |
| `model_not_found` | The model was retired. Change `model=` in `main.py` (see Configuration notes). |
| `DATABASE_URL is not set` / connection errors | Make sure Postgres is running and `DATABASE_URL` in `.env` is correct. |
| PowerShell: "running scripts is disabled" | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\python.exe -m streamlit run frontend.py`. |

## Tech stack

[LangGraph](https://github.com/langchain-ai/langgraph) · [LangChain Groq](https://python.langchain.com/docs/integrations/chat/groq/) · [Streamlit](https://streamlit.io/) · [Aviationstack](https://aviationstack.com/) · [Tavily](https://tavily.com/) · [PostgreSQL](https://www.postgresql.org/) · [psycopg 3](https://www.psycopg.org/psycopg3/)
