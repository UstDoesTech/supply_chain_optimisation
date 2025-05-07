# Databricks notebook source
# MAGIC %md
# MAGIC # Context-Aware vs Traditional Inventory Simulation Notebook
# MAGIC
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC <img src="https://www.advancinganalytics.co.uk/hubfs/AdvancingAnalytics-2024/images/logo/Advancing-Analytics-Logo-2024.svg" alt="Advancing Analytics" style="width: 100px; height: 100px">
# MAGIC </div>
# MAGIC
# MAGIC This notebook lets you simulate and compare the performance of traditional and context-aware inventory optimisation strategies. It supports external signal customisation and visualizations. Feel free to use your own data and external datasets, or use the example data provided. 
# MAGIC
# MAGIC If you're running this on a Unity Catalog enabled workspace, you can explore the resulting models interactively through the accompanying AI/BI dashhboard and Genie Space.

# COMMAND ----------

dbutils.widgets.text("CatalogName", "analytics_sandbox", "Name of UC Catalog") # hive_metastore if non-UC workspace
dbutils.widgets.text("DatabaseName", "default", "Name of Database in Catalog")

catalog_name = dbutils.widgets.get("CatalogName")
database_name = dbutils.widgets.get("DatabaseName")

print(catalog_name)
print(database_name)

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {database_name}")

# COMMAND ----------

from supply_chain_optimisation.main import *

# COMMAND ----------

train_cutoff = '2021-07-01'
test_cutoff = '2022-01-01'
categories = ['Electronics', 'Clothing', 'Food', 'Home Goods', 'Health']

# Add category-specific buffer
buffer_dict = {
    'Electronics': 0.15,
    'Clothing': 0.10,
    'Food': 0.20,
    'Home Goods': 0.10,
    'Health': 0.05
}

feature_cols = [
        'weekofyear', 'month', 'dayofweek', 'lead_time', 
        'lag_demand', 'lag_demand_2', 'lag_demand_3', 'demand_ma_4',
        'promo_flag', 'holiday_flag'
    ]


# COMMAND ----------

# Define External Signals
def promo_flag_logic():
    return when(rand() < 0.1, 1).otherwise(0)

def holiday_flag_logic():
    return when(col('month').isin([11, 12]), 1).otherwise(0)

# COMMAND ----------

product_info_df, product_info_pd = get_product_info()
supply_chain_df = generate_demand_data(product_info_pd)  # Ensure generate_demand_data is defined or imported

# Display dataset information
print(f"Supply Chain Dataset: {supply_chain_df.count()} observations")
print(f"Date Range: {supply_chain_df.agg(min('date'), max('date')).collect()[0]}")
print(f"Products: {supply_chain_df.select('product_id').distinct().count()}")
print(f"Categories: {supply_chain_df.select('category').distinct().count()}")

product_info_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{database_name}.product_info")
supply_chain_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{database_name}.supply_chain")

# COMMAND ----------

# Split dataset
train_df, test_df = split_dataframe_train_test(train_cutoff, test_cutoff, supply_chain_df)

train_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{database_name}.training_set")
test_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{database_name}.testing_set")

# COMMAND ----------

# Calculate traditional inventory parameters
product_params = get_product_params(train_df, product_info_df)
display(product_params.select(
    'product_id', 'category', 'demand_avg', 'lead_time_avg', 
    'eoq', 'safety_stock', 'reorder_point'
))

# COMMAND ----------

traditional_plan = generate_traditional_plan(test_df, product_params)
display(traditional_plan)

traditional_plan.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{database_name}.traditional_plan")

# COMMAND ----------

train_df_ml = get_train_df_ml(train_df)
train_df_ml = add_column_with_logic(train_df_ml, 'promo_flag', promo_flag_logic)
train_df_ml = add_column_with_logic(train_df_ml, 'holiday_flag', holiday_flag_logic)
models = train_context_aware_model(train_df_ml, feature_cols, categories)

# COMMAND ----------

test_df_ml = prepare_test_data(test_df)

# COMMAND ----------

test_df_ml = add_column_with_logic(test_df_ml, 'promo_flag', promo_flag_logic)
test_df_ml = add_column_with_logic(test_df_ml, 'holiday_flag', holiday_flag_logic)

# COMMAND ----------

context_aware_predictions = get_context_aware_forecast(categories, test_df_ml, models)

# COMMAND ----------

context_aware_plan = generate_context_aware_plan(context_aware_predictions, product_params, buffer_dict)
display(context_aware_plan)

context_aware_plan.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{database_name}.context_aware_plan")

# COMMAND ----------

traditional_simulation = simulate_traditional_approach(traditional_plan, product_info_df)
display(traditional_simulation)

traditional_simulation.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{database_name}.traditional_simulation")

# COMMAND ----------

context_aware_simulation = simulate_context_aware_approach(context_aware_plan, product_info_df)
display(context_aware_simulation)

context_aware_simulation.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{database_name}.context_aware_simulation")

# COMMAND ----------

traditional_metrics, context_aware_metrics = get_performance_metrics(traditional_simulation, context_aware_simulation)
print(traditional_metrics)
print(context_aware_metrics)

# COMMAND ----------

comparison_df = get_comparison(traditional_metrics, context_aware_metrics)
display(comparison_df)

comparison_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{database_name}.plan_comparison")

# COMMAND ----------

category_comparison = get_category_comparison(traditional_simulation, context_aware_simulation)
display(category_comparison)

category_comparison.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{database_name}.category_comparison")

# COMMAND ----------

accuracy_df = get_forecast_accuracy(traditional_plan, context_aware_plan)
display(accuracy_df)

accuracy_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{database_name}.forecast_accuracy")

# COMMAND ----------

product_comparison = get_product_comparison(traditional_simulation, context_aware_simulation, product_info_df)
display(product_comparison)

product_comparison.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{database_name}.product_comparison")