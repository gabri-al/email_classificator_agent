# Databricks notebook source
# MAGIC %md
# MAGIC # Register & Deploy Model
# MAGIC Register model on UC model registry and deploy it as Model Serving Endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ## Env & Variables
# MAGIC Running package install and defining global variables from separate Notebooks.

# COMMAND ----------

# MAGIC %run ./step_0_Environment

# COMMAND ----------

# MAGIC %run ./step_0_Variables

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Agent definition to a .py file

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC from step_3_Agent_Graph_inMemory import react_email_classifier
# MAGIC
# MAGIC # Build the LangGraph agent here
# MAGIC def build_agent():
# MAGIC     return react_email_classifier
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC react_email_classifier_agent = build_agent()
# MAGIC mlflow.models.set_model(react_email_classifier_agent)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure MLflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from agent import react_email_classifier_agent

# COMMAND ----------


