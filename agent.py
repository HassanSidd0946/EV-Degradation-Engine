# agent.py

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from typing import TypedDict
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize the model correctly for Azure
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    api_version=AZURE_API_VERSION,
)


class BatteryState(TypedDict):
    predictions   : list
    actual        : list
    mae           : float
    final_soh     : float
    analysis      : str
    recommendation: str


def analyze_degradation(state: BatteryState) -> BatteryState:
    prompt = f"""
    You are an EV battery expert. Analyze this data:
    - Final SOH: {state['final_soh']}%
    - MAE: {state['mae']:.4f} Ah
    - Total cycles: {len(state['predictions'])}
    - Started at: {state['predictions'][0]:.3f} Ah
    - Ended at: {state['predictions'][-1]:.3f} Ah
    Give 2-3 line technical analysis with specific numbers.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    state['analysis'] = response.content
    return state

def give_recommendation(state: BatteryState) -> BatteryState:
    prompt = f"""
    Battery SOH is {state['final_soh']}%.
    - Above 85%: healthy
    - 70-85%: caution
    - Below 70%: replace soon
    Give one specific recommendation in 2 lines.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    state['recommendation'] = response.content
    return state

def build_battery_agent():
    graph = StateGraph(BatteryState)
    graph.add_node("analyze",   analyze_degradation)
    graph.add_node("recommend", give_recommendation)
    graph.set_entry_point("analyze")
    graph.add_edge("analyze",   "recommend")
    graph.add_edge("recommend", END)
    return graph.compile()

battery_agent = build_battery_agent()