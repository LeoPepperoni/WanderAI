from crewai import Agent
from textwrap import dedent
from langchain_openai import ChatOpenAI
from tools.search_tools import SearchTools
from tools.calculator_tools import CalculatorTools


class TravelAgents:
    def __init__(self):
        try:
            self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        except Exception as e:
            print(f"Error initializing gpt-3.5-turbo: {e}")

        try:
            self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        except Exception as e:
            print(f"Error initializing gpt-4: {e}")

    def expert_travel_agent(self):
        try:
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
                tools=[SearchTools.search_internet, CalculatorTools.calculate],
                allow_delegation=False,
                verbose=True,
                llm=self.OpenAIGPT4,
            )
        except Exception as e:
            print(f"Error creating Expert Travel Agent: {e}")

    def city_selection_expert(self):
        try:
            return Agent(
                role="City Selection Expert",
                backstory=dedent(f"""Expert at analyzing travel data to pick ideal destinations"""),
                goal=dedent(f"""
                            Select the best cities to travel to based on weather, season, prices and travelers 
                            interests
                            """),
                tools=[SearchTools.search_internet],
                allow_delegation=False,
                verbose=True,
                llm=self.OpenAIGPT4,
            )
        except Exception as e:
            print(f"Error creating City Selection Expert: {e}")

    def local_tour_guide(self):
        try:
            return Agent(
                role="Local Tour Guide",
                backstory=dedent(f"""
                                Knowledgeable local guide with extensive information 
                                about the city it's attractions and customs
                                """),
                goal=dedent(f"""Provide the BEST insights about the selected city"""),
                tools=[SearchTools.search_internet],
                allow_delegation=False,
                verbose=True,
                llm=self.OpenAIGPT4,
            )
        except Exception as e:
            print(f"Error creating Local Tour Guide: {e}")
