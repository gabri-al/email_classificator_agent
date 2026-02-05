# Email Classificator Agent

This agent is an improved and advanced version of the email classifier presented [here](medium.com/towards-artificial-intelligence/llm-powered-email-classification-on-databricks-2089cdae4806).
In this new version, a custom agent is built to support a customer service department with email classification.

## Repo Structure
This Repo contains the code needed to build the custom agent on Databricks, using LangGraph:
`
email_classificator_agent/
├─ step0_Environment: Notebook used to install packages and configure the environment
├─ step0_Variables: Notebook containing global variables defining resources in the Databricks Worskpace, such as: catalog, schema, Lakebase instance, LLM endpoint, MLflow experiment location
├─ step1_Data_Generation: This Notebook is used to generate mock up datasets (customers, orders, tickets and new emails), persist them on Unity Catalog and create retrieve tools (as SQL UDF) that allow the anget to retrieve content from a synced Lakebase db
├─ step2_Prompts_Definition: All prompts for the nodes and tools used by the agent are defined in this Notebook.
├─ step3_Agent_Graph_inMemory: Implementation of the LangGraph components of the agent (state, nodes, edges) to run in memory.
├─ step4_Tests: In this Notebook, the agent is used on the mock up dataset, using MLflow Tracing
├─ step5_DeployModel: This Notebook register the agent as an MLflow model on Unity Catalog, wrapping the agent in a [ResponsesAgent](https://mlflow.org/docs/latest/genai/serving/responses-agent/) Pyfunc class to deploy it on a Model Serving Endpoint
├─ requirements.txt: versioned dependencies
`
## LangGraph graph
This agent leverages ReAct plus tool calling to classify emails and produce an output following a desired schema:

## Example
Here's an example from the mock up dataset produced by the agent, hosted on a Databricks Model Serving and using Playground as UI:

