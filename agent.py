from step_3_Agent_Graph_inMemory import react_email_classifier

# Build the LangGraph agent here
def build_agent():
    return react_email_classifier

mlflow.langchain.autolog()
react_email_classifier_agent = build_agent()
mlflow.models.set_model(react_email_classifier_agent)
