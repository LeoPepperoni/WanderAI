from crewai import Agent
from textwrap import dedent
from langchain.llms import OpenAI, Ollama
from langchain_openai import ChatOpenAI

# Creating Agents Cheat Sheet:
# - Think backwards from a goal and think which employee you need to hire to get the job done.
# - Define a leader of the crew who orients the other agents towards the goal.
# - Define which experts the leader needs to communicate with and delegate tasks to.
#
# Goal:
# - Create a 7-day travel itinerary with detailed per-day plans, including budget, packing suggestions and safety tips


# Leader:
# - Expert Travel Agent

# Employees to hire:
# - City Selection Expert
# - Local Tour Guide

# Notes:
# - Agents should be results driven and have a clear goal in mind
# - Role is their job title
# - goals should be actionable
# - Backstory should be their resume





# This is an example of how to define custom agents.
# You can define as many agents as you want.
# You can also define custom tasks in tasks.py
class CustomAgents:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        self.Ollama = Ollama(model="openhermes")

    def expert_travel_agent(self):
        return Agent(
            role="Expert Travel Agent",
            backstory=dedent(f"""
            Expert in travel planning and logistics. 
            I have decades of experience making travel itineraries
            """),
            goal=dedent(f"""
                        Create a 7-day travel itinerary with detailed per-day plans, 
                        including budget, packing suggestions and safety tips
                        """),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def city_selection_expert(self):
        return Agent(
            role="City Selection Expert",
            backstory=dedent(f"""Expert at analyzing travel data to pick ideal destinations"""),
            goal=dedent(f"""
                        Select the best cities to travel to based on weather, season, prices and travelers 
                        interests
                        """),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def local_tour_guide(self):
        return Agent(
            role="Local Tour Guide",
            backstory=dedent(f"""
                            Knowledgeable local guide with extensive information 
                            about the city it's attractions and customs
                            """),
            goal=dedent(f"""Provide the BEST insights about the selected city"""),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )
