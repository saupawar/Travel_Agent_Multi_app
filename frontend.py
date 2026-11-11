"""Streamlit front end for the Travel Multi-Agent app.

Run from the project root:
    streamlit run frontend.py
"""

import json
import os
import re
from datetime import datetime

import streamlit as st
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="TripPilot · AI Travel Planner", page_icon="✈️", layout="wide")


# ---------- Copy Streamlit Cloud secrets into environment variables ----------
REQUIRED_KEYS = ["GROQ_API_KEY", "AVIATIONSTACK_API_KEY", "tavily_API_KEY"]  # DATABASE_URL is optional


def load_secrets_into_env() -> str | None:
    """On Streamlit Cloud, keys live in st.secrets. main.py and the tools read
    os.environ, so copy them across (works for top-level keys or keys inside a
    [section]). Locally, with no secrets.toml, .env is used instead.

    Returns a description of why secrets couldn't be read, or None."""
    try:
        items = dict(st.secrets)
    except FileNotFoundError:  # no secrets file (normal for local runs)
        return "no secrets file was found"
    except Exception as exc:  # usually invalid TOML in the Secrets box
        return f"the Secrets box could not be read ({type(exc).__name__}: {exc})"
    if not items:
        return "the Secrets box is empty"
    for key, value in items.items():
        if isinstance(value, dict) or hasattr(value, "items"):
            for sub_key, sub_value in dict(value).items():
                os.environ.setdefault(sub_key, str(sub_value))
        else:
            os.environ.setdefault(key, str(value))
    return None


# ---------- Load the compiled LangGraph app once (shared DB connection) ----------
@st.cache_resource(show_spinner="Connecting to the travel agents…")
def load_app():
    secrets_problem = load_secrets_into_env()
    from dotenv import load_dotenv
    load_dotenv()  # local runs: fill anything still missing from .env

    missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        found = []
        try:
            found = list(st.secrets.keys())
        except Exception:
            pass
        raise RuntimeError(
            "Missing settings: " + ", ".join(missing) + ". "
            + (f"Reason: {secrets_problem}. " if secrets_problem else "")
            + (f"Secret names found: {', '.join(found)}. " if found else "")
            + "On Streamlit Cloud open ⋮ → Settings → Secrets and use TOML format with quotes, "
            'e.g.  GROQ_API_KEY = "gsk_..."  then Save and Reboot the app.'
        )

    from main import app  # importing main opens the Postgres connection once
    return app


# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1200px; }
    .hero {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        color: white; padding: 1.6rem 2rem; border-radius: 18px; margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: white; }
    .hero p  { margin: .35rem 0 0; opacity: .92; font-size: 1.05rem; }
    .card {
        border: 1px solid rgba(128,128,128,.25); border-radius: 14px;
        padding: 1rem 1.2rem; margin-bottom: .9rem; background: rgba(128,128,128,.04);
    }
    .card h4 { margin: 0 0 .35rem; }
    .muted { opacity: .7; font-size: .88rem; }
    .route { display:flex; align-items:center; gap:1rem; justify-content:space-between; }
    .route .end { flex:1; }
    .route .end.right { text-align:right; }
    .route .code { font-size:1.6rem; font-weight:700; line-height:1.1; }
    .route .line { flex:1; text-align:center; opacity:.6; font-size:1.2rem; }
    .badge {
        display:inline-block; padding:.15rem .6rem; border-radius:999px;
        font-size:.78rem; font-weight:600; text-transform:capitalize;
    }
    .s-scheduled { background:#dbeafe; color:#1e40af; }
    .s-active    { background:#dcfce7; color:#166534; }
    .s-landed    { background:#e0e7ff; color:#3730a3; }
    .s-delayed, .s-incident, .s-diverted { background:#fef3c7; color:#92400e; }
    .s-cancelled { background:#fee2e2; color:#991b1b; }
    .s-unknown   { background:#e5e7eb; color:#374151; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Helpers ----------
def parse_json_list(raw: str) -> list[dict]:
    try:
        data = json.loads(raw) if raw else []
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def fmt_time(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%d %b, %H:%M")
    except ValueError:
        return iso


def esc(text) -> str:
    """Minimal HTML escape for values placed inside our card markup."""
    return (str(text) if text is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def first_error(items: list[dict]) -> str | None:
    if items and "error" in items[0]:
        return items[0]["error"]
    return None


def render_flight(f: dict):
    airline, dep, arr = f.get("airline", {}), f.get("departure", {}), f.get("arrival", {})
    status = (f.get("status") or "unknown").lower()
    delay = dep.get("delay_min")
    dep_extra = " · ".join(
        x for x in [f"T{dep['terminal']}" if dep.get("terminal") else "", f"Gate {dep['gate']}" if dep.get("gate") else ""] if x
    )
    arr_extra = " · ".join(
        x for x in [f"T{arr['terminal']}" if arr.get("terminal") else "", f"Gate {arr['gate']}" if arr.get("gate") else ""] if x
    )
    st.markdown(
        f"""
        <div class="card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.6rem">
            <h4>✈️ {esc(airline.get('name') or 'Unknown airline')}
              <span class="muted">&nbsp;{esc(airline.get('flight_number') or '')}</span></h4>
            <span class="badge s-{esc(status)}">{esc(status)}</span>
          </div>
          <div class="route">
            <div class="end">
              <div class="code">{esc(dep.get('iata') or '—')}</div>
              <div class="muted">{esc(dep.get('airport') or '')}</div>
              <div><b>{fmt_time(dep.get('scheduled'))}</b></div>
              <div class="muted">{esc(dep_extra)}</div>
            </div>
            <div class="line">──────── ✈ ────────</div>
            <div class="end right">
              <div class="code">{esc(arr.get('iata') or '—')}</div>
              <div class="muted">{esc(arr.get('airport') or '')}</div>
              <div><b>{fmt_time(arr.get('scheduled'))}</b></div>
              <div class="muted">{esc(arr_extra)}</div>
            </div>
          </div>
          {f'<div class="muted" style="margin-top:.5rem">⏱ Departure delayed by {esc(delay)} min</div>' if delay else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hotel(h: dict):
    snippet = re.sub(r"\s+", " ", h.get("content") or "").strip()
    if len(snippet) > 260:
        snippet = snippet[:260].rsplit(" ", 1)[0] + "…"
    score = h.get("score")
    with st.container(border=True):
        st.markdown(f"#### 🏨 {h.get('title') or 'Untitled result'}")
        if snippet:
            st.write(snippet)
        c1, c2 = st.columns([3, 1])
        if h.get("url"):
            c1.link_button("View details ↗", h["url"])
        if isinstance(score, (int, float)):
            c2.caption(f"Relevance {score:.0%}")


AGENT_STEPS = {
    "flight_agent": "✈️ Searching flights",
    "hotel_agent": "🏨 Finding hotels",
    "itinerary_agent": "🗺️ Drafting your itinerary",
    "final_agent": "✨ Putting it all together",
}


def plan_trip(query: str, thread_id: str) -> dict:
    """Stream the graph so the user sees each agent finish in real time."""
    app = load_app()
    state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "flight_results": "",
        "hotel_results": "",
        "itinerary": "",
        "final_response": "",
        "llm_calls": 0,
    }
    config = {"configurable": {"thread_id": thread_id}}
    result = dict(state)

    with st.status("Your travel agents are on it…", expanded=True) as status:
        for update in app.stream(state, config=config, stream_mode="updates"):
            for node, changes in update.items():
                st.write(f"{AGENT_STEPS.get(node, node)} — done ✅")
                for key, value in (changes or {}).items():
                    if key != "messages":
                        result[key] = value
        status.update(label="Your trip is ready!", state="complete", expanded=False)
    return result


# ---------- Session state ----------
st.session_state.setdefault("history", [])   # list of {"query", "result", "time"}
st.session_state.setdefault("selected", None)
st.session_state.setdefault("prefill", "")


# ---------- Sidebar ----------
with st.sidebar:
    st.header("🧳 Traveller")
    traveller = st.text_input("Your name", value="aarohi", help="Your trips are saved under this name.")
    thread_id = f"user_{re.sub(r'[^a-z0-9_]', '', traveller.lower()) or 'guest'}"

    st.divider()
    st.subheader("💡 Try an example")
    examples = [
        "Flights from Mumbai to Dubai and good hotels near Dubai Marina",
        "New York to London trip, show flights and central hotels",
        "Delhi to Singapore, family-friendly hotels and a 3-day plan",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.prefill = ex

    if st.session_state.history:
        st.divider()
        st.subheader("🕘 Recent trips")
        for i, item in enumerate(reversed(st.session_state.history)):
            idx = len(st.session_state.history) - 1 - i
            if st.button(f"{item['time']} · {item['query'][:38]}…", key=f"hist_{idx}", use_container_width=True):
                st.session_state.selected = idx

    st.divider()
    st.caption("Powered by LangGraph · Groq · Aviationstack · Tavily")


# ---------- Header + search ----------
st.markdown(
    """
    <div class="hero">
      <h1>✈️ TripPilot</h1>
      <p>Tell us where you're going — our AI agents find flights, hotels and build your itinerary.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("trip_form", clear_on_submit=False):
    query = st.text_area(
        "Where would you like to go?",
        value=st.session_state.prefill,
        placeholder="e.g. Flights from Mumbai to Dubai next week and hotels near the beach",
        height=90,
    )
    submitted = st.form_submit_button("🔎 Plan my trip", type="primary", use_container_width=True)

if submitted:
    if not query.strip():
        st.warning("Please describe your trip first — include where you're flying from and to.")
    else:
        try:
            result = plan_trip(query.strip(), thread_id)
            st.session_state.history.append(
                {"query": query.strip(), "result": result, "time": datetime.now().strftime("%H:%M")}
            )
            st.session_state.selected = len(st.session_state.history) - 1
            st.session_state.prefill = ""
        except Exception as exc:  # show a friendly message instead of a stack trace
            st.error("Sorry — something went wrong while planning your trip.")
            with st.expander("Technical details"):
                st.exception(exc)


# ---------- Results ----------
sel = st.session_state.selected
if sel is None or sel >= len(st.session_state.history):
    st.info("👆 Describe your trip above, or pick an example from the sidebar to get started.")
    st.stop()

trip = st.session_state.history[sel]
result = trip["result"]
flights = parse_json_list(result.get("flight_results", ""))
hotels = parse_json_list(result.get("hotel_results", ""))
flight_err, hotel_err = first_error(flights), first_error(hotels)

st.markdown(f"### Your trip: *{trip['query']}*")

m1, m2, m3 = st.columns(3)
m1.metric("Flights found", 0 if flight_err else len(flights))
m2.metric("Hotel options", 0 if hotel_err else len(hotels))
m3.metric("Agent steps", result.get("llm_calls", 0))

tab_overview, tab_flights, tab_hotels, tab_plan = st.tabs(
    ["✨ Overview", "✈️ Flights", "🏨 Hotels", "🗺️ Itinerary"]
)

with tab_overview:
    st.markdown(result.get("final_response") or "_No summary was generated._")
    st.download_button(
        "⬇️ Download trip plan (.md)",
        data=result.get("final_response", ""),
        file_name="trip_plan.md",
        mime="text/markdown",
    )

with tab_flights:
    if flight_err:
        st.warning(f"Couldn't load flights: {flight_err}")
        st.caption("Tip: mention both cities or airport codes, e.g. “from BOM to DXB”.")
    elif not flights:
        st.info("No flights were found for this route right now.")
    else:
        statuses = sorted({(f.get("status") or "unknown") for f in flights})
        chosen = st.multiselect("Filter by status", statuses, default=statuses)
        for f in flights:
            if (f.get("status") or "unknown") in chosen:
                render_flight(f)

with tab_hotels:
    if hotel_err:
        st.warning(f"Couldn't load hotels: {hotel_err}")
    elif not hotels:
        st.info("No hotel results were found.")
    else:
        cols = st.columns(2)
        for i, h in enumerate(hotels):
            with cols[i % 2]:
                render_hotel(h)

with tab_plan:
    st.markdown(result.get("itinerary") or "_No itinerary was generated._")
