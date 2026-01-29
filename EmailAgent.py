
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Variable definition
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

## Variables pointing to Unity Catalog
catalog_ = 'users'
schema_ = 'gabriele_albini'
lakebase_instance_ = 'gabriele-lb'
llm_endpoint = 'databricks-claude-sonnet-4-5'

Assistant_Prompt = (
  """
    You are a customer support assistant, tasked with classifying incoming emails and suggesting following actions to improve customer relationship.\n\n

    Use the available tools to:\n
    * (When applicable) Retrieve the relevant context information to enrich the customer's email. If the required information to use a tool cannot be obtained or if any retrieval tool generates an error, assume that the related context is not available.\n
    * Classify the email by assigning a label.\n
    * Suggest the next steps.\n\n
    
    *Important*: If the initial request doesn't contain both the sender email address and the email body, return `Others` as final label, DO NOT call any other tool and ask the user to provide the required information.

    Once you've obtained a label for the email, STOP calling tools.
  """)

Classification_Prompt = (
  """
    You are an email dispatcher assistant.\n
    Based on the available context, your tasks are:\n
        (1) Assign a label to the customer's email.\n
        (2) Provide a summarized explanation of why you assigned such label.\n
        (3) Recommend next steps, based on the context.\n\n

    Here are the available labels you can choose from, together with a description of when to use them:\n
    * `Order Issues`: For emails about missing items, wrong orders, delayed shipments, tracking problems or any problem related to orders.\n
    * `Returns & Refunds`: For requests or questions about return procedures, refund status, or exchange policies.\n
    * `Claims & Complaints`: For defect reports, damaged goods, or any service dissatisfaction.\n
    * `Account & Data Requests`: For customers asking to update personal information, manage account settings, or exercise data rights (e.g., GDPR requests).\n
    * `Product Information`: For inquiries about product details, availability, sizing, or compatibility.\n
    * `Payment & Billing`: For issues related to charges, payment methods, invoice requests, or failed transactions.\n
    * `Spam`: for emails that are suspected of being a spam, not referring to any plausible customer's request or follow ups.\n
    * `Others`: for emails that cannot be related to any of the previous labels.\n\n

    Here is the contextual information.\n

    The body email sent by the customer:\n
    `{email_body}`\n

    The customer's information present in our database:\n
    `{customer_info}`\n

    The most recent customer's order present in our database:\n
    `{order_info}`\n

    The most recent customer's ticket present in our database:\n
    `{ticket_info}`\n\n

    *IMPORTANT*: Consider the following exceptions when choosing a label:
    * If the email is a follow up to a previous ticket (e.g., the customer is sending a reply; the customer is acknowledging information;), assign the same label as the previous ticket.
    
    Now classify the email, explain your reasoning and recommend next steps.
  """)

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

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Base LLM
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
from databricks_langchain import ChatDatabricks
model = ChatDatabricks(endpoint = llm_endpoint, temperature=0) 

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Respond Model (LLM + Output schema)
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
model_with_output_schema = model.with_structured_output(FinalOutput)


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

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Create Tool Binding
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

tools = uc_tools + [classify_email] ## Combine all tools above (uc + custom)
model_with_tools = model.bind_tools(tools)

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

import mlflow
mlflow.models.set_model(react_email_classifier)
