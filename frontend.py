"""Streamlit front end for the Travel Multi-Agent app.

Run from the project root:
    streamlit run frontend.py
"""

import json
import os
import re
from datetime import datetime
from urllib.parse import quote_plus, urlparse

import streamlit as st
from langchain_core.messages import HumanMessage

from user_session import is_valid_thread_id, name_from_thread_id, new_guest_id, thread_id_from_name

st.set_page_config(page_title="TripPilot · AI Travel Planner", page_icon="✈️", layout="wide")


# ---------- Copy Streamlit Cloud secrets into environment variables ----------
REQUIRED_KEYS = ["GROQ_API_KEY", "AVIATIONSTACK_API_KEY", "tavily_API_KEY", "DATABASE_URL"]


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
    .muted { opacity: .68; font-size: .86rem; }

    /* Trip summary bar */
    .tripbar {
        display:flex; flex-wrap:wrap; align-items:center; gap:.5rem 1.4rem;
        border:1px solid rgba(128,128,128,.25); border-radius:14px;
        padding:.85rem 1.2rem; margin:.4rem 0 1rem; background:rgba(99,102,241,.06);
    }
    .tripbar .route { font-size:1.25rem; font-weight:700; }
    .tripbar .chip  { font-size:.9rem; opacity:.85; }

    /* Flight result row */
    .fl-row {
        display:grid; grid-template-columns: 1.4fr 2.2fr 1.2fr 1.2fr; gap:1rem; align-items:center;
        border:1px solid rgba(128,128,128,.25); border-radius:14px;
        padding:1rem 1.2rem; margin-bottom:.75rem; background:rgba(128,128,128,.03);
        transition: box-shadow .15s, border-color .15s;
    }
    .fl-row:hover { border-color:#6366f1; box-shadow:0 2px 10px rgba(99,102,241,.15); }
    .fl-airline { display:flex; align-items:center; gap:.7rem; }
    .fl-logo {
        width:42px; height:42px; border-radius:10px; flex:none;
        display:flex; align-items:center; justify-content:center;
        font-weight:800; font-size:.95rem; color:#fff;
    }
    .fl-name { font-weight:600; line-height:1.2; }
    .fl-times { font-size:1.35rem; font-weight:700; letter-spacing:.2px; }
    .fl-times sup { font-size:.7rem; color:#dc2626; font-weight:700; margin-left:2px; }
    .fl-dur { font-weight:600; }
    .fl-stops { font-size:.86rem; color:#16a34a; font-weight:600; }
    .fl-act { text-align:right; }
    .fl-act a {
        display:inline-block; margin-top:.45rem; padding:.42rem .9rem; border-radius:999px;
        background:#6366f1; color:#fff !important; text-decoration:none; font-weight:600; font-size:.86rem;
    }
    .fl-act a:hover { background:#4f46e5; }
    .fl-tags { margin-bottom:.15rem; }
    .tag { display:inline-block; font-size:.72rem; font-weight:700; padding:.12rem .5rem;
           border-radius:6px; margin-right:.3rem; }
    .tag-early { background:#e0f2fe; color:#075985; }
    .tag-fast  { background:#dcfce7; color:#166534; }
    @media (max-width: 760px) {
        .fl-row { grid-template-columns: 1fr 1fr; }
        .fl-act { text-align:left; }
    }

    .badge {
        display:inline-block; padding:.15rem .6rem; border-radius:999px;
        font-size:.75rem; font-weight:600; text-transform:capitalize;
    }
    .s-scheduled { background:#dbeafe; color:#1e40af; }
    .s-active    { background:#dcfce7; color:#166534; }
    .s-landed    { background:#e0e7ff; color:#3730a3; }
    .s-delayed, .s-incident, .s-diverted { background:#fef3c7; color:#92400e; }
    .s-cancelled { background:#fee2e2; color:#991b1b; }
    .s-unknown   { background:#e5e7eb; color:#374151; }

    /* Stay (hotel) result row */
    .st-row {
        display:grid; grid-template-columns: 150px 1fr 150px; gap:1.1rem; align-items:stretch;
        border:1px solid rgba(128,128,128,.25); border-radius:14px; overflow:hidden;
        margin-bottom:.8rem; background:rgba(128,128,128,.03);
    }
    .st-row:hover { border-color:#6366f1; box-shadow:0 2px 10px rgba(99,102,241,.15); }
    .st-thumb { display:flex; align-items:center; justify-content:center; font-size:2.6rem; min-height:130px; }
    .st-body { padding:.9rem 0; }
    .st-title { font-weight:700; font-size:1.05rem; margin-bottom:.2rem; }
    .st-src { font-size:.82rem; opacity:.7; margin-bottom:.4rem; }
    .st-snip { font-size:.9rem; opacity:.9; }
    .st-side { padding:.9rem 1rem .9rem 0; display:flex; flex-direction:column;
               justify-content:space-between; align-items:flex-end; text-align:right; }
    .score { display:flex; align-items:center; gap:.45rem; }
    .score b { background:#16a34a; color:#fff; border-radius:8px; padding:.2rem .45rem; font-size:.95rem; }
    .score span { font-size:.82rem; font-weight:600; }
    .st-side a {
        padding:.42rem .9rem; border-radius:999px; background:#6366f1; color:#fff !important;
        text-decoration:none; font-weight:600; font-size:.86rem;
    }
    @media (max-width: 760px) {
        .st-row { grid-template-columns: 1fr; }
        .st-thumb { min-height:80px; }
        .st-body, .st-side { padding:0 1rem .9rem; align-items:flex-start; text-align:left; }
    }
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


def esc(text) -> str:
    """Minimal HTML escape for values placed inside our markup."""
    return (str(text) if text is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def first_error(items: list[dict]) -> str | None:
    if items and "error" in items[0]:
        return items[0]["error"]
    return None


def leg_dt(leg: dict) -> datetime | None:
    """Local wall-clock datetime of a leg (Aviationstack: local time, no conversion)."""
    try:
        return datetime.fromisoformat(leg.get("scheduled") or "").replace(tzinfo=None)
    except ValueError:
        return None


def fmt_clock(dt: datetime | None) -> str:
    return dt.strftime("%H:%M") if dt else "—"


def fmt_day(dt: datetime | None) -> str:
    return dt.strftime("%a, %d %b %Y") if dt else "—"


def fmt_duration(minutes) -> str:
    if not isinstance(minutes, (int, float)) or minutes <= 0:
        return "—"
    h, m = divmod(int(minutes), 60)
    return f"{h}h {m:02d}m"


def short_airport(name: str | None) -> str:
    """'Chhatrapati Shivaji International (Sahar International)' -> 'Chhatrapati Shivaji International'."""
    return re.sub(r"\s*\(.*?\)", "", name or "").strip()


LOGO_COLOURS = ["#2563eb", "#7c3aed", "#db2777", "#ea580c", "#059669", "#0891b2", "#4f46e5", "#b45309"]


def airline_logo(airline: dict) -> str:
    code = (airline.get("iata") or (airline.get("name") or "?")[:2]).upper()[:3]
    colour = LOGO_COLOURS[sum(map(ord, code)) % len(LOGO_COLOURS)]
    return f'<div class="fl-logo" style="background:{colour}">{esc(code)}</div>'


def fares_link(f: dict) -> str:
    dep, arr = f.get("departure", {}), f.get("arrival", {})
    q = f"Flights from {dep.get('iata', '')} to {arr.get('iata', '')} on {f.get('flight_date', '')}"
    return "https://www.google.com/travel/flights?q=" + quote_plus(q)


def time_bucket(dt: datetime | None) -> str:
    if not dt:
        return "Unknown"
    h = dt.hour
    if h < 6:
        return "Early morning (00:00–05:59)"
    if h < 12:
        return "Morning (06:00–11:59)"
    if h < 18:
        return "Afternoon (12:00–17:59)"
    return "Evening (18:00–23:59)"


def render_flight(f: dict, tags: list[str]):
    airline, dep, arr = f.get("airline", {}), f.get("departure", {}), f.get("arrival", {})
    status = (f.get("status") or "unknown").lower()
    d_dt, a_dt = leg_dt(dep), leg_dt(arr)
    offset = f.get("day_offset") or 0
    plus = f"<sup>+{offset}</sup>" if offset > 0 else ""
    stops = f.get("stops", 0)
    stops_txt = "Nonstop" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}"
    operated = airline.get("operated_by")
    operated_txt = (
        f'<div class="muted">Operated by {esc(operated)}'
        f'{" (" + esc(airline.get("operating_flight")) + ")" if airline.get("operating_flight") else ""}</div>'
        if operated else ""
    )
    delay = dep.get("delay_min")
    delay_txt = f'<div class="muted" style="color:#b45309">Delayed {esc(delay)} min</div>' if delay else ""
    tag_html = "".join(
        f'<span class="tag {"tag-early" if t == "Earliest" else "tag-fast"}">{esc(t)}</span>' for t in tags
    )

    st.markdown(
        f"""
        <div class="fl-row">
          <div class="fl-airline">
            {airline_logo(airline)}
            <div>
              <div class="fl-name">{esc(airline.get('name') or 'Unknown airline')}</div>
              <div class="muted">{esc(airline.get('flight_number') or '')}</div>
              {operated_txt}
            </div>
          </div>
          <div>
            <div class="fl-tags">{tag_html}</div>
            <div class="fl-times">{fmt_clock(d_dt)} – {fmt_clock(a_dt)}{plus}</div>
            <div class="muted">{esc(short_airport(dep.get('airport')))} ({esc(dep.get('iata'))})
              – {esc(short_airport(arr.get('airport')))} ({esc(arr.get('iata'))})</div>
            <div class="muted">{fmt_day(d_dt)}{' → ' + fmt_day(a_dt) if offset else ''}</div>
          </div>
          <div>
            <div class="fl-dur">{fmt_duration(f.get('duration_min'))}</div>
            <div class="fl-stops">{stops_txt}</div>
          </div>
          <div class="fl-act">
            <span class="badge s-{esc(status)}">{esc(status)}</span>
            {delay_txt}
            <div><a href="{esc(fares_link(f))}" target="_blank" rel="noopener">Check fares ↗</a></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"Flight details · {airline.get('flight_number') or ''}"):
        c1, c2 = st.columns(2)
        for col, label, leg, dt in ((c1, "🛫 Departure", dep, d_dt), (c2, "🛬 Arrival", arr, a_dt)):
            col.markdown(f"**{label}**")
            col.markdown(
                f"{leg.get('airport') or '—'} ({leg.get('iata') or '—'})  \n"
                f"**{fmt_day(dt)} · {fmt_clock(dt)}** local time"
                + (f" ({leg['timezone']})" if leg.get("timezone") else "") + "  \n"
                f"Terminal {leg.get('terminal') or '—'} · Gate {leg.get('gate') or '—'}"
                + (f"  \nEstimated: {leg['estimated_time']}" if leg.get("estimated_time") else "")
                + (f"  \nActual: {leg['actual_time']}" if leg.get("actual_time") else "")
            )


def match_label(score: float) -> str:
    if score >= 0.9:
        return "Excellent match"
    if score >= 0.75:
        return "Great match"
    if score >= 0.6:
        return "Good match"
    return "Related"


THUMB_GRADIENTS = [
    "linear-gradient(135deg,#0ea5e9,#6366f1)", "linear-gradient(135deg,#f59e0b,#ef4444)",
    "linear-gradient(135deg,#10b981,#0ea5e9)", "linear-gradient(135deg,#8b5cf6,#ec4899)",
    "linear-gradient(135deg,#14b8a6,#84cc16)",
]


def render_stay(h: dict, i: int):
    snippet = re.sub(r"\s+", " ", h.get("content") or "").strip()
    snippet = re.sub(r"[|#*]+", " ", snippet)  # strip table/markdown noise from scraped text
    if len(snippet) > 220:
        snippet = snippet[:220].rsplit(" ", 1)[0] + "…"
    url = h.get("url") or ""
    source = urlparse(url).netloc.replace("www.", "") if url else ""
    score = h.get("score")
    score_html = (
        f'<div class="score"><span>{match_label(score)}</span><b>{score * 10:.1f}</b></div>'
        if isinstance(score, (int, float)) else "<div></div>"
    )
    link_html = f'<a href="{esc(url)}" target="_blank" rel="noopener">View ↗</a>' if url else ""
    st.markdown(
        f"""
        <div class="st-row">
          <div class="st-thumb" style="background:{THUMB_GRADIENTS[i % len(THUMB_GRADIENTS)]}">🏨</div>
          <div class="st-body">
            <div class="st-title">{esc(h.get('title') or 'Untitled result')}</div>
            <div class="st-src">🌐 {esc(source)}</div>
            <div class="st-snip">{esc(snippet)}</div>
          </div>
          <div class="st-side">
            {score_html}
            {link_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def restore_trips(thread_id: str, limit: int = 10) -> list[dict]:
    """Rebuild this traveller's completed trips from the Postgres checkpoints,
    so their context survives page reloads and new browser sessions."""
    try:
        app = load_app()
        trips = []
        for snap in app.get_state_history({"configurable": {"thread_id": thread_id}}):
            values = snap.values or {}
            if snap.next or not values.get("final_response"):
                continue  # only the final checkpoint of each finished trip
            try:
                when = datetime.fromisoformat(snap.created_at).astimezone().strftime("%d %b %H:%M")
            except (TypeError, ValueError):
                when = "saved"
            trips.append({"query": values.get("user_query", ""), "result": dict(values), "time": when})
            if len(trips) >= limit:
                break
        return list(reversed(trips))  # oldest first, like new trips are appended
    except Exception:
        return []  # no database / not configured yet: start with an empty history


# ---------- Session state ----------
st.session_state.setdefault("history", [])   # list of {"query", "result", "time"}
st.session_state.setdefault("selected", None)
st.session_state.setdefault("prefill", "")

# Traveller identity: ?tid=... in the URL keeps the same session across reloads
_url_tid = st.query_params.get("tid")
_url_tid = _url_tid if is_valid_thread_id(_url_tid) else None
if "guest_id" not in st.session_state:
    st.session_state.guest_id = (
        _url_tid if _url_tid and _url_tid.startswith("guest_") else new_guest_id()
    )
st.session_state.setdefault("traveller_name", name_from_thread_id(_url_tid))


def start_new_session():
    st.session_state.guest_id = new_guest_id()
    st.session_state.traveller_name = ""


# ---------- Sidebar ----------
with st.sidebar:
    st.header("🧳 Traveller")
    traveller = st.text_input(
        "Your name (optional)",
        key="traveller_name",
        placeholder="e.g. Priya",
        help="Enter a name to save your trips under it. Leave empty to use an automatic guest session.",
    )
    thread_id = thread_id_from_name(traveller) or st.session_state.guest_id
    if st.query_params.get("tid") != thread_id:
        st.query_params["tid"] = thread_id

    if traveller.strip():
        st.caption(f"Trips are saved for **{traveller.strip()}** · session `{thread_id}`")
    else:
        st.caption(f"Guest session `{thread_id}`  \nBookmark this page to come back to your trips.")
    st.button("🔄 Start a new session", on_click=start_new_session, use_container_width=True)

# Switching traveller/session: load that traveller's saved trips
if st.session_state.get("active_thread") != thread_id:
    st.session_state.active_thread = thread_id
    st.session_state.history = restore_trips(thread_id)
    st.session_state.selected = len(st.session_state.history) - 1 if st.session_state.history else None

with st.sidebar:
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
flights = [] if flight_err else flights
hotels = [] if hotel_err else hotels

# Trip summary bar: "Mumbai (BOM) → Dubai (DXB) · Wed, 11 Nov 2026 · 5 flights · 5 stays"
if flights:
    d0, a0 = flights[0].get("departure", {}), flights[0].get("arrival", {})
    route = (f"{esc(short_airport(d0.get('airport')))} ({esc(d0.get('iata'))}) → "
             f"{esc(short_airport(a0.get('airport')))} ({esc(a0.get('iata'))})")
    dates = sorted({fmt_day(leg_dt(f.get("departure", {}))) for f in flights} - {"—"})
    when = dates[0] if len(dates) == 1 else (f"{dates[0]} – {dates[-1]}" if dates else "")
else:
    route, when = esc(trip["query"]), ""
st.markdown(
    f"""
    <div class="tripbar">
      <span class="route">✈️ {route}</span>
      {f'<span class="chip">📅 {when}</span>' if when else ''}
      <span class="chip">🛫 {len(flights)} flight{'s' if len(flights) != 1 else ''}</span>
      <span class="chip">🏨 {len(hotels)} stay{'s' if len(hotels) != 1 else ''}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_flights, tab_stays, tab_plan, tab_summary = st.tabs(
    ["✈️ Flights", "🏨 Stays", "🗺️ Itinerary", "✨ Trip summary"]
)

with tab_flights:
    if flight_err:
        st.warning(f"Couldn't load flights: {flight_err}")
        st.caption("Tip: mention both cities or airport codes, e.g. “from BOM to DXB”.")
    elif not flights:
        st.info("No flights were found for this route right now.")
    else:
        filt, res = st.columns([1, 3], gap="large")

        with filt:
            st.markdown("**Sort by**")
            sort_by = st.selectbox(
                "Sort by",
                ["Recommended", "Departure (earliest)", "Departure (latest)",
                 "Arrival (earliest)", "Duration (shortest)"],
                label_visibility="collapsed",
            )
            st.markdown("**Airlines**")
            airlines = sorted({(f.get("airline", {}).get("name") or "Unknown") for f in flights})
            chosen_air = [a for a in airlines if st.checkbox(a, value=True, key=f"air_{sel}_{a}")]
            st.markdown("**Departure time**")
            buckets = sorted({time_bucket(leg_dt(f.get("departure", {}))) for f in flights})
            chosen_bkt = [b for b in buckets if st.checkbox(b, value=True, key=f"bkt_{sel}_{b}")]
            st.markdown("**Status**")
            statuses = sorted({(f.get("status") or "unknown") for f in flights})
            chosen_st = [s for s in statuses if st.checkbox(s.capitalize(), value=True, key=f"st_{sel}_{s}")]

        shown = [
            f for f in flights
            if (f.get("airline", {}).get("name") or "Unknown") in chosen_air
            and time_bucket(leg_dt(f.get("departure", {}))) in chosen_bkt
            and (f.get("status") or "unknown") in chosen_st
        ]
        far_future, far_long = datetime.max, 10**9
        sorters = {
            "Departure (earliest)": lambda f: leg_dt(f.get("departure", {})) or far_future,
            "Departure (latest)": lambda f: leg_dt(f.get("departure", {})) or datetime.min,
            "Arrival (earliest)": lambda f: leg_dt(f.get("arrival", {})) or far_future,
            "Duration (shortest)": lambda f: f.get("duration_min") or far_long,
        }
        if sort_by in sorters:
            shown.sort(key=sorters[sort_by], reverse=(sort_by == "Departure (latest)"))

        # Highlight tags, like booking sites' "Earliest" / "Fastest" labels
        with_dt = [f for f in flights if leg_dt(f.get("departure", {}))]
        earliest = min(with_dt, key=lambda f: leg_dt(f["departure"])) if with_dt else None
        with_dur = [f for f in flights if f.get("duration_min")]
        fastest = min(with_dur, key=lambda f: f["duration_min"]) if with_dur else None

        with res:
            st.markdown(f"#### Choose your flight  \n<span class='muted'>{len(shown)} of {len(flights)} "
                        "results · times are local to each airport · fares open in Google Flights</span>",
                        unsafe_allow_html=True)
            if not shown:
                st.info("No flights match these filters.")
            for f in shown:
                tags = []
                if f is earliest and len(flights) > 1:
                    tags.append("Earliest")
                if f is fastest and len(flights) > 1:
                    tags.append("Fastest")
                render_flight(f, tags)

with tab_stays:
    if hotel_err:
        st.warning(f"Couldn't load stays: {hotel_err}")
    elif not hotels:
        st.info("No stays were found.")
    else:
        st.markdown(f"#### {len(hotels)} places to stay  \n<span class='muted'>From web search · "
                    "score shows how well each result matches your request</span>", unsafe_allow_html=True)
        for i, h in enumerate(sorted(hotels, key=lambda h: h.get("score") or 0, reverse=True)):
            render_stay(h, i)

with tab_plan:
    st.markdown(result.get("itinerary") or "_No itinerary was generated._")

with tab_summary:
    st.markdown(result.get("final_response") or "_No summary was generated._")
    st.download_button(
        "⬇️ Download trip plan (.md)",
        data=result.get("final_response", ""),
        file_name="trip_plan.md",
        mime="text/markdown",
    )
