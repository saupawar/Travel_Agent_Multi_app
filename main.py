import os
import json
from typing import TypedDict, Annotated, Optional
import operator

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import get_flights


load_dotenv()  # reads GROQ_API_KEY from .env

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,     # steadier tool calls
    max_retries=2,
)

DATABASE_URL = os.getenv("DATABASE_URL")

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results:str
    hotel_results:str
    itinerary: str
    final_response: str
    llm_calls: int


# ---------- Helper: pull route + destination out of the user query ----------
class TripDetails(BaseModel):
    dep_iata: Optional[str] = Field(None, description="3-letter IATA code of the departure airport")
    arr_iata: Optional[str] = Field(None, description="3-letter IATA code of the arrival airport")
    destination_city: Optional[str] = Field(None, description="Destination city name")


trip_extractor = llm.with_structured_output(TripDetails)


def extract_trip(user_query: str) -> TripDetails:
    return trip_extractor.invoke(
        "Extract the trip details from this travel request. Convert city names to "
        "their main airport's 3-letter IATA code (e.g. New York -> JFK, Mumbai -> BOM). "
        "Use null for anything not mentioned.\n\n"
        f"Request: {user_query}"
    )


# ---------- Node 1: flights (user_query -> get_flights -> flight_results) ----------
def flight_node(state: TravelState) -> dict:
    trip = extract_trip(state["user_query"])

    if trip.dep_iata and trip.arr_iata:
        flights = get_flights(trip.dep_iata.upper(), trip.arr_iata.upper(), limit=5)
    else:
        flights = [{"error": "Could not work out both departure and arrival airports."}]

    flight_results = json.dumps(flights, indent=2)  # state field is a str
    return {
        "flight_results": flight_results,
        "messages": [AIMessage(content=f"Flight search results:\n{flight_results}", name="flight_agent")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ---------- Node 2: hotels (user_query -> tavily_search -> hotel_results) ----------
def hotel_node(state: TravelState) -> dict:
    hotels = tavily_search(f"best hotels for: {state['user_query']}", max_results=5)
    hotel_results = json.dumps(hotels, indent=2)
    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content=f"Hotel search results:\n{hotel_results}", name="hotel_agent")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ---------- Node 3: itinerary (day-by-day plan from flights + hotels) ----------
def itinerary_node(state: TravelState) -> dict:
    response = llm.invoke([
        SystemMessage(content=(
            "You are a travel planner. Using the flight and hotel data provided, "
            "draft a concise day-by-day itinerary for the trip (arrival day, main days, "
            "departure day). Base timings on the flight data when available. "
            "Do not invent flight numbers, prices or hotel names that are not in the data."
        )),
        HumanMessage(content=(
            f"User request: {state['user_query']}\n\n"
            f"Flight results:\n{state.get('flight_results', '')}\n\n"
            f"Hotel results:\n{state.get('hotel_results', '')}"
        )),
    ])
    return {
        "itinerary": response.content,
        "messages": [AIMessage(content=response.content, name="itinerary_agent")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ---------- Node 4: final agent (polished travel response for the user) ----------
def final_node(state: TravelState) -> dict:
    response = llm.invoke([
        SystemMessage(content=(
            "You are a friendly travel assistant writing the final reply to the user. "
            "Using ONLY the data provided, respond in Markdown with these sections:\n"
            "1. **Trip Summary** - one or two sentences answering the user's request.\n"
            "2. **Flights** - a table: Airline & Flight No. | Departure (airport, time, "
            "terminal/gate) | Arrival (airport, time) | Status.\n"
            "3. **Hotels** - top options as bullet points with a one-line note and link.\n"
            "4. **Itinerary** - the day-by-day plan, tidied up.\n"
            "5. **Tips** - 2-3 practical tips for this trip.\n"
            "If flight or hotel data contains an error or is empty, say so in that "
            "section and suggest what the user can try. Never invent data."
        )),
        HumanMessage(content=(
            f"User request: {state['user_query']}\n\n"
            f"Flight results:\n{state.get('flight_results', '')}\n\n"
            f"Hotel results:\n{state.get('hotel_results', '')}\n\n"
            f"Draft itinerary:\n{state.get('itinerary', '')}"
        )),
    ])
    return {
        "final_response": response.content,
        "messages": [AIMessage(content=response.content, name="final_agent")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ---------- Graph ----------
def build_graph(checkpointer=None):
    graph = StateGraph(TravelState)
    graph.add_node("flight_agent", flight_node)
    graph.add_node("hotel_agent", hotel_node)
    graph.add_node("itinerary_agent", itinerary_node)
    graph.add_node("final_agent", final_node)

    graph.add_edge(START, "flight_agent")
    graph.add_edge("flight_agent", "hotel_agent")
    graph.add_edge("hotel_agent", "itinerary_agent")
    graph.add_edge("itinerary_agent", "final_agent")
    graph.add_edge("final_agent", END)
    return graph.compile(checkpointer=checkpointer)


# ---------- Persistent connection so both CLI and Streamlit can share the compiled app ----------
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

# PostgresSaver needs autocommit (for setup) and dict rows (to read checkpoints)
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    prepare_threshold=0,
    row_factory=dict_row,
)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()  # creates checkpoint tables on first run

app = build_graph(checkpointer)


def run(user_query: str, thread_id: str = "user_aarohi") -> dict:
    """Run one travel request. Same thread_id = same saved conversation."""
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(
        {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )


if __name__ == "__main__":
    user_input = input("Enter travel request: ")

    result = run(user_input, thread_id="user_aarohi")

    print("\nFINAL RESPONSE:\n")
    print(result["final_response"])
    print(f"\n(LLM calls: {result['llm_calls']})")
