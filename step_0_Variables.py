# Databricks notebook source
# MAGIC %md
# MAGIC # Global Variables definition
# MAGIC Creating global variables.

# COMMAND ----------

## Variables pointing to Unity Catalog
catalog_ = 'users'
schema_ = 'gabriele_albini'
lakebase_instance_ = 'gabriele-lb'

# COMMAND ----------

## Variables to recreate dummy data on UC and Lakebase
regenerate_data = False
regenerate_lakebase_data = False

# COMMAND ----------

## LLM: Variable containing the LLM Endpoint to use
llm_endpoint = 'databricks-gpt-5-1'
llm_endpoint = 'databricks-claude-sonnet-4-5'

# COMMAND ----------

## Variable for MLflow
current_user_email = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_dir_ = f'/Users/{current_user_email}/'
