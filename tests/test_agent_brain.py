from app.agents.agent_controller import AgentController

agent = AgentController()

print(agent.decide("Explain AI"))
print(agent.decide("Explain previous answer"))
print(agent.decide("What is in this image?"))

