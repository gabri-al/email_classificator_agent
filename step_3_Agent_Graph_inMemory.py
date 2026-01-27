# Databricks notebook source
# MAGIC %md
# MAGIC # LangGraph Agent in Memory
# MAGIC Defining the agent's graph (state, nodes, edges) and run it in memory.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Env & Variables
# MAGIC Running package install and defining global variables from separate Notebooks.

# COMMAND ----------

# %run ./step_0_Environment

# COMMAND ----------

# MAGIC %run ./step_0_Variables

# COMMAND ----------

# MAGIC %run ./step_2_Prompts_Definition

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schemas
# MAGIC Defining the desired agent's State and output schemas.

# COMMAND ----------

# DBTITLE 1,Final Output Schema
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Output Schema
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

from pydantic import BaseModel, Field
from typing import Literal

class FinalOutput(BaseModel):
  Customer_Email: str = Field(description = "Email of the customer who sent the message.")
  Customer_Id: str = Field(description = "Id of the customer, related to the email. If not found, return NULL.")
  Customer_Context: str = Field(description = "Summary of all the retrieved information related to the customer's request. If nothing was retrieved, return NULL.")
  Label: Literal[
      "Order Issues",
      "Returns & Refunds",
      "Claims & Complaints",
      "Account & Data Requests",
      "Product Information",
      "Payment & Billing",
      "Spam",
      "Others",
      "Classification Error"] = Field(
    description="Generated label assigned to the email",
    default="Classification Error")
  Rationale: str = Field(description="Reasoning behind the label choice")
  Next_steps: str = Field(description="Recommended action items based on the customer's email")


# COMMAND ----------

# DBTITLE 1,Graph State
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Graph State
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages, AnyMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

## Set up the agent state to concatenate messages in memory
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    final_output: FinalOutput | None

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLMs
# MAGIC Defining the base LLM to be used by the agent nodes.

# COMMAND ----------

# DBTITLE 1,Base LLM
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Base LLM
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
from databricks_langchain import ChatDatabricks
model = ChatDatabricks(endpoint = llm_endpoint, temperature=0) 

# COMMAND ----------

# DBTITLE 1,Respond Model (LLM + Output schema)
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Respond Model (LLM + Output schema)
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
model_with_output_schema = model.with_structured_output(FinalOutput)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tools
# MAGIC Specify all tools the agent can use to retrieve information and classify emails.

# COMMAND ----------

# DBTITLE 1,Classification Tool
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Classification Tool
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

from langchain_core.tools import tool
from langgraph.graph import MessagesState

@tool
def classify_email(customer_info: str, order_info: str, ticket_info: str,  email_body: str) -> str:
    """ Classify the customer's email into one of the predefined labels, based on the email_body and the available context you could retrieve:\n
    * customer_info - representing the customer's details in our database\n
    * order_info - representing the latest customer's order\n
    * ticket_info - representing the latest customer's ticket\n
    """
    
    # Promopt template
    MODEL_SYSTEM_MESSAGE = Classification_Prompt

    # Pass data to the prompt
    full_prompt = MODEL_SYSTEM_MESSAGE.format(
        email_body=email_body,
        customer_info=customer_info,
        order_info=order_info,
        ticket_info=ticket_info
    )

    # Invoke LLM & return
    result = model.invoke([full_prompt])
    return result.content

# COMMAND ----------

# DBTITLE 1,UC functions as Retrieval Tools
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# UC functions as Retrieval Tools
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
from databricks_langchain import UCFunctionToolkit

toolkit = UCFunctionToolkit(function_names=[
      f"{catalog_}.{schema_}.classificator_agent_customer_retriever",
      f"{catalog_}.{schema_}.classificator_agent_order_retriever",
      f"{catalog_}.{schema_}.classificator_agent_ticket_retriever",
])
uc_tools = toolkit.tools

# COMMAND ----------

# DBTITLE 1,Create Tool Binding
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Create Tool Binding
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

tools = uc_tools + [classify_email] ## Combine all tools above (uc + custom)
model_with_tools = model.bind_tools(tools)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Nodes
# MAGIC Creating the reasoning node (assistant) and the respond node.

# COMMAND ----------

# DBTITLE 1,Reasoning Node (Assistant)
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Reasoning Node (Assistant)
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

# System message
sys_msg = SystemMessage(content=Assistant_Prompt)

# Reasoning Node
def assistant(state: AgentState) -> AgentState:
  result = model_with_tools.invoke([sys_msg] + state["messages"])
  return {"messages": state["messages"] + [result]}

# COMMAND ----------

# DBTITLE 1,Respond Node
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Respond Node: Enforce output schema on the final answer
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def respond(state: AgentState) -> AgentState:
    # Uses the last message as input to the structured output model
    last_msg = state["messages"][-1]
    result = model_with_output_schema.invoke(
        [HumanMessage(content=last_msg.content)]
    )
    return {"final_output": result}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Graph

# COMMAND ----------

# DBTITLE 1,ReAct Graph
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# ReAct Graph
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display

# Graph
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("respond", respond)

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
    {"tools": "tools", "__end__": "respond"} # customize mapping which would route to end by default
)
builder.add_edge("tools", "assistant") # ReAct: return tool outputs to reason on it!
builder.add_edge("respond", END)
react_email_classifier = builder.compile() # No checkpointer (memory) needed

# COMMAND ----------

# DBTITLE 1,Display the Graph
# Display the Graph
display(Image(react_email_classifier.get_graph().draw_mermaid_png()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing
# MAGIC Run a test case with Notebook context

# COMMAND ----------

# DBTITLE 1,Import test cases
## Save test cases
spark.sql("USE CATALOG "+catalog_)
spark.sql("USE SCHEMA "+schema_)
df_test_emails = spark.table("classificator_agent_emails")

# COMMAND ----------

# DBTITLE 1,Pick Test case
from pyspark.sql.functions import *

# Extract test case from the dataframe with examples
id_ = 7
test_record = df_test_emails.filter(col("Email_Id") == id_).first()
message_ = f"We received from: {test_record.Email_Sender} the following email: {test_record.Email_Body}"
display(message_)

# COMMAND ----------

# DBTITLE 1,Launch Agent
from pyspark.sql.functions import *
from langchain_core.messages import HumanMessage

# Invoke the Agent
config_ = {"configurable": {"thread_id": id_}} # Not required if we're not using memory

mlflow.langchain.autolog()
with mlflow.start_run(run_name="React Email Classifier Agent - Test"):
  request = [
    HumanMessage(content = message_)
  ]
  messages = react_email_classifier.invoke({"messages": request}, config_)

  for m in messages['messages']:
      m.pretty_print()

# COMMAND ----------

# DBTITLE 1,Available Context: Customer Details
# MAGIC %sql
# MAGIC -- Available Context: Customer Details
# MAGIC SELECT classificator_agent_customer_retriever('sofia.rossi@example.com') AS customer_details;

# COMMAND ----------

# DBTITLE 1,Available Context: Order
# MAGIC %sql
# MAGIC -- Available Context: Order Details
# MAGIC SELECT classificator_agent_order_retriever(3) AS order_details;

# COMMAND ----------

# DBTITLE 1,Available Context: Ticket
# MAGIC %sql
# MAGIC -- Available Context: Ticket Details
# MAGIC SELECT classificator_agent_ticket_retriever(3) AS ticket_details;
