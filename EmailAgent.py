
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
# MLflow PyFunc Wrapper for Unity Catalog Registration
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

import mlflow
import os
from typing import Any, Dict, List

class EmailClassifierAgent(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for the LangGraph email classification agent.
    This class enables the agent to be registered in Unity Catalog and deployed
    as a model serving endpoint with conversational agent format.
    """
    
    def load_context(self, context):
        """
        Called when the model is loaded. Initializes the LangGraph agent with UC tools.
        """
        from databricks_langchain import UCFunctionToolkit
        from databricks.sdk import WorkspaceClient
        from langgraph.graph import START, StateGraph, END
        from langgraph.prebuilt import tools_condition, ToolNode
        
        # Initialize Databricks client for UC function access
        # In model serving, use environment variables for authentication
        workspace_url = os.environ.get('DATABRICKS_HOST')
        token = os.environ.get('DATABRICKS_TOKEN')
        
        if workspace_url and token:
            w = WorkspaceClient(host=workspace_url, token=token)
        else:
            # Fallback to default authentication (for local testing)
            w = WorkspaceClient()
        
        # Initialize UC Function Toolkit with client
        toolkit = UCFunctionToolkit(
            function_names=[
                f"{catalog_}.{schema_}.classificator_agent_customer_retriever",
                f"{catalog_}.{schema_}.classificator_agent_order_retriever",
                f"{catalog_}.{schema_}.classificator_agent_ticket_retriever",
            ],
            client=w
        )
        uc_tools = toolkit.tools
        
        # Combine all tools
        tools = uc_tools + [classify_email]
        model_with_tools = model.bind_tools(tools)
        
        # System message
        sys_msg = SystemMessage(content=Assistant_Prompt)
        
        # Reasoning Node
        def assistant(state: AgentState) -> AgentState:
            result = model_with_tools.invoke([sys_msg] + state["messages"])
            return {"messages": state["messages"] + [result]}
        
        # Respond Node: Enforce output schema on the final answer
        def respond(state: AgentState) -> AgentState:
            last_msg = state["messages"][-1]
            result = model_with_output_schema.invoke(
                [HumanMessage(content=last_msg.content)]
            )
            return {"final_output": result}
        
        # Build the graph
        builder = StateGraph(AgentState)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(tools))
        builder.add_node("respond", respond)
        
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
            {"tools": "tools", "__end__": "respond"}
        )
        builder.add_edge("tools", "assistant")
        builder.add_edge("respond", END)
        
        self.agent = builder.compile()
    
    def predict(self, context, model_input, params=None):
        """
        Predict method for conversational agent format.
        
        Args:
            context: MLflow context (not used)
            model_input: Dict or List of dicts with 'messages' key containing conversation history
            params: Optional parameters (not used)
            
        Returns:
            Dict with classification results in the final_output field
        """
        # Handle both single dict and list of dicts
        if isinstance(model_input, dict):
            inputs = [model_input]
        else:
            inputs = model_input
        
        results = []
        for input_data in inputs:
            # Get messages from input
            messages = input_data.get('messages', [])
            
            # If messages is a string, convert to message format
            if isinstance(messages, str):
                messages = [HumanMessage(content=messages)]
            elif isinstance(messages, list) and len(messages) > 0:
                # Convert dict messages to LangChain message objects if needed
                converted_messages = []
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        if role == 'user':
                            converted_messages.append(HumanMessage(content=content))
                        elif role == 'assistant':
                            converted_messages.append(AIMessage(content=content))
                        else:
                            converted_messages.append(HumanMessage(content=content))
                    else:
                        converted_messages.append(msg)
                messages = converted_messages
            else:
                messages = [HumanMessage(content="No input provided")]
            
            # Invoke the agent
            response = self.agent.invoke({"messages": messages})
            
            # Extract the final output
            final_output = response.get('final_output')
            
            if final_output:
                result = {
                    'Customer_Email': final_output.Customer_Email,
                    'Customer_Id': final_output.Customer_Id,
                    'Customer_Context': final_output.Customer_Context,
                    'Label': final_output.Label,
                    'Rationale': final_output.Rationale,
                    'Next_steps': final_output.Next_steps
                }
            else:
                result = {
                    'Customer_Email': 'Unknown',
                    'Customer_Id': 'NULL',
                    'Customer_Context': 'NULL',
                    'Label': 'Classification Error',
                    'Rationale': 'Agent failed to produce output',
                    'Next_steps': 'Manual review required'
                }
            
            results.append(result)
        
        # Return single dict if single input, otherwise list
        return results[0] if len(results) == 1 else results

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Load Model Function for MLflow Registration
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def load_model():
    """
    Factory function to create an instance of the EmailClassifierAgent.
    This function is used during model registration with MLflow.
    """
    return EmailClassifierAgent()
