# Import required libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkContext
from databricks.sdk.runtime import *

# spark = SparkSession.builder.appName("SupplyChainOptimization").getOrCreate()

# Explicitly import the Python min function
import builtins

# Generate 3-year supply chain dataset with realistic patterns and disruptions
np.random.seed(42)

# Date range: 2019-2022 (weekly data)
dates = pd.date_range(start='2019-01-01', end='2022-01-01', freq='W')

# Create baseline for 5 product categories with different characteristics
categories = ['Electronics', 'Clothing', 'Food', 'Home Goods', 'Health']
products_per_category = 5
total_products = len(categories) * products_per_category

# Product base demand (weekly)
base_demand = np.random.lognormal(mean=5.0, sigma=1.0, size=total_products)

# Lead times (in weeks) - different for each product
base_lead_times = np.random.uniform(1, 8, total_products)

# Cost parameters
unit_costs = np.random.uniform(10, 100, total_products)
holding_cost_rate = 0.25  # 25% annual holding cost
stockout_cost_multiplier = np.random.uniform(1.5, 5, total_products)  # Cost multiplier for stockouts

# Create product information dataframe first as Pandas, then convert to Spark
def get_product_info():
    product_info_pd = pd.DataFrame({
        'product_id': range(total_products),
        'category': [cat for cat in categories for _ in range(products_per_category)],
        'base_demand': base_demand,
        'base_lead_time': base_lead_times,
        'unit_cost': unit_costs,
        'stockout_cost_multiplier': stockout_cost_multiplier
    })

    # Convert to Spark DataFrame
    product_info_df = spark.createDataFrame(product_info_pd)
    return product_info_df, product_info_pd

def generate_demand_data(product_info_pd):
    # Generate time series data for each product
    demand_data = []
    time_idx = np.arange(len(dates))

    for product_id in range(total_products):
        category = product_info_pd.loc[product_id, 'category']
        base = product_info_pd.loc[product_id, 'base_demand']
        lead_base = product_info_pd.loc[product_id, 'base_lead_time']
        
        # Create product-specific patterns
        # 1. Seasonality - different by category
        if category == 'Electronics':
            # Holiday peak, summer low
            seasonality = 0.3 * np.sin(2 * np.pi * (time_idx - 45) / 52)
        elif category == 'Clothing':
            # Seasonal changes (spring/fall peaks)
            seasonality = 0.2 * np.sin(4 * np.pi * time_idx / 52) + 0.1 * np.sin(2 * np.pi * time_idx / 52)
        elif category == 'Food':
            # Holiday peak, summer peak
            seasonality = 0.15 * np.sin(2 * np.pi * time_idx / 52) + 0.15 * np.sin(2 * np.pi * (time_idx - 26) / 52)
        elif category == 'Home Goods':
            # Spring peak, fall low
            seasonality = 0.25 * np.sin(2 * np.pi * (time_idx - 13) / 52)
        else:  # Health
            # Winter peak (cold/flu season)
            seasonality = 0.3 * np.cos(2 * np.pi * time_idx / 52)
        
        # 2. Trend - different by product
        trend_factor = np.random.uniform(-0.1, 0.2)  # Some declining, most growing
        trend = trend_factor * time_idx / len(time_idx)
        
        # 3. COVID-19 Effect (starts at week 60, peaks at week 75, gradually diminishes)
        covid_start = 60  # March 2020
        covid_effect = np.zeros_like(time_idx, dtype=float)
        
        covid_mask = time_idx >= covid_start
        covid_time = time_idx[covid_mask] - covid_start
        
        if category == 'Electronics':
            # Increased demand during lockdowns
            covid_effect[covid_mask] = 0.5 * np.exp(-0.05 * (covid_time - 15)**2)
        elif category == 'Clothing':
            # Decreased then recovered
            covid_effect[covid_mask] = -0.4 * np.exp(-0.05 * covid_time) + 0.2 * np.exp(-0.05 * (covid_time - 40)**2)
        elif category == 'Food':
            # Panic buying then new normal
            covid_effect[covid_mask] = 0.8 * np.exp(-0.1 * covid_time) + 0.2
        elif category == 'Home Goods':
            # Delayed peak (home improvement during lockdowns)
            covid_effect[covid_mask] = 0.6 * np.exp(-0.03 * (covid_time - 25)**2)
        else:  # Health
            # Sustained increase
            covid_effect[covid_mask] = 0.7 * (1 - np.exp(-0.1 * covid_time))
        
        # 4. Supply chain disruption effect on lead times (increases after COVID starts)
        lead_disruption = np.zeros_like(time_idx, dtype=float)
        lead_disruption[covid_mask] = 0.5 * (1 - np.exp(-0.05 * covid_time))
        
        # Combine all effects for demand
        demand = base * (1 + seasonality + trend + covid_effect)
        
        # Add noise to demand
        noise_level = 0.15  # 15% noise
        noise = np.random.normal(0, noise_level * base, len(time_idx))
        demand = np.maximum(0, demand + noise)
        
        # Generate lead times with disruption effect
        lead_time = lead_base * (1 + lead_disruption)
        
        # Add noise to lead times
        lead_noise = np.random.normal(0, 0.1 * lead_base, len(time_idx))
        lead_time = np.maximum(1, lead_time + lead_noise)
        
        # Add to dataset
        for week, date in enumerate(dates):
            demand_data.append({
                'date': date,
                'product_id': product_id,
                'category': category,
                'demand': demand[week],
                'lead_time': lead_time[week]
            })

    # Create Spark DataFrame from the demand data
    # First convert to pandas, then to Spark for efficiency
    demand_data_pd = pd.DataFrame(demand_data)
    supply_chain_df = spark.createDataFrame(demand_data_pd)

    return supply_chain_df

def split_dataframe_train_test(train_cutoff, test_cutoff, supply_chain_df):
    # Create train and test datasets
    train_df = supply_chain_df.filter(col('date') < train_cutoff)
    test_df = supply_chain_df.filter((col('date') >= train_cutoff) & (col('date') < test_cutoff))

    # Display split information
    print(f"Training set: {train_df.count()} observations")
    train_dates = train_df.agg(min('date'), max('date')).collect()[0]
    print(f"Training date range: {train_dates[0]} to {train_dates[1]}")

    print(f"Test set: {test_df.count()} observations")
    test_dates = test_df.agg(min('date'), max('date')).collect()[0]
    print(f"Test date range: {test_dates[0]} to {test_dates[1]}")

    return train_df, test_df

@udf(returnType=DoubleType())
def calculate_eoq(demand, unit_cost, holding_cost_rate, order_cost=50):
    """Calculate Economic Order Quantity"""
    annual_demand = demand * 52  # Convert weekly to annual
    annual_holding_cost = unit_cost * holding_cost_rate
    return float(np.sqrt((2 * annual_demand * order_cost) / annual_holding_cost))

# Safety stock calculation UDF
@udf(returnType=DoubleType())
def calculate_safety_stock(demand_std, lead_time_avg, service_level=0.95):
    """Calculate safety stock level"""
    z_score = norm.ppf(service_level)
    return float(z_score * demand_std * np.sqrt(lead_time_avg))

# Reorder point calculation UDF
@udf(returnType=DoubleType())
def calculate_reorder_point(demand_avg, lead_time_avg, safety_stock):
    """Calculate reorder point"""
    return float(demand_avg * lead_time_avg + safety_stock)

# Calculate traditional inventory parameters
# First calculate statistics for each product
def get_product_params (train_df, product_info_df):
    product_stats = train_df.groupBy('product_id').agg(
        avg('demand').alias('demand_avg'),
        stddev('demand').alias('demand_std'),
        avg('lead_time').alias('lead_time_avg'),
        stddev('lead_time').alias('lead_time_std')
    )

    # Join with product info
    product_params = product_stats.join(product_info_df, 'product_id')

    # Calculate inventory parameters
    product_params = product_params.withColumn(
        'eoq', calculate_eoq('demand_avg', 'unit_cost', lit(holding_cost_rate))
    )

    product_params = product_params.withColumn(
        'safety_stock', calculate_safety_stock('demand_std', 'lead_time_avg', lit(0.95))
    )

    product_params = product_params.withColumn(
        'reorder_point', calculate_reorder_point('demand_avg', 'lead_time_avg', 'safety_stock')
    )

    return product_params

# Generate traditional inventory plan
def generate_traditional_plan(test_df, product_params):
    traditional_plan = test_df.join(
        product_params.select(
            'product_id', 'demand_avg', 'eoq', 'safety_stock', 'reorder_point'
        ),
        'product_id'
    )

    # Calculate required inventory and order quantity
    traditional_plan = traditional_plan.withColumn(
        'forecast_demand', col('demand_avg')
    ).withColumn(
        'required_inventory', col('forecast_demand') * col('lead_time') + col('safety_stock')
    ).withColumn(
        'order_quantity', when(col('required_inventory') > 0, col('eoq')).otherwise(0)
    )

    # Select relevant columns
    traditional_plan = traditional_plan.select(
        'date', 'product_id', 'category', 'forecast_demand', 'demand',
        'lead_time', 'safety_stock', 'reorder_point', 'order_quantity', 'required_inventory'
    ).withColumnRenamed('demand', 'actual_demand')

    return traditional_plan

def get_train_df_ml(train_df):
    # Add time-based features
    train_df_ml = train_df.withColumn('weekofyear', weekofyear('date'))
    train_df_ml = train_df_ml.withColumn('month', month('date'))
    train_df_ml = train_df_ml.withColumn('dayofweek', dayofweek('date'))

    # Create window for lagged features
    window_spec = Window.partitionBy('product_id').orderBy('date')

    # Add lagged features
    train_df_ml = train_df_ml.withColumn('lag_demand', lag('demand', 1).over(window_spec))
    train_df_ml = train_df_ml.withColumn('lag_demand_2', lag('demand', 2).over(window_spec))
    train_df_ml = train_df_ml.withColumn('lag_demand_3', lag('demand', 3).over(window_spec))

    # Add moving averages
    train_df_ml = train_df_ml.withColumn(
        'demand_ma_4', avg('demand').over(window_spec.rowsBetween(-4, -1))
    )

    # # Add external signals
    # train_df_ml = train_df_ml.withColumn(
    #     'promo_flag', when(rand() < 0.1, 1).otherwise(0)
    # )
    # train_df_ml = train_df_ml.withColumn(
    #     'holiday_flag', when(col('month').isin([11, 12]), 1).otherwise(0)
    # )

    # Drop rows with null values (due to lag)
    train_df_ml = train_df_ml.na.drop()

    return train_df_ml

def train_context_aware_model(train_df_ml, feature_cols, categories):
    # Check if all feature columns exist in the DataFrame
    missing_cols = [col for col in feature_cols if col not in train_df_ml.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Create vector assembler
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

    # Train category-specific models using Spark ML
    models = {}

    # Train a model for each category
    for category in categories:
        # Filter data for this category
        category_data = train_df_ml.filter(col('category') == category)
        
        gbt = GBTRegressor(featuresCol='features', labelCol='demand', maxDepth=5, maxIter=10)
        pipeline = Pipeline(stages=[assembler, gbt])
        
        # Train model
        model = pipeline.fit(category_data)
        models[category] = model

    # Display the models dictionary to ensure models are trained for each category
    for category, model in models.items():
        print(f"Category: {category}, Model: {model}")

    return models

def prepare_test_data(test_df):
    # Prepare test data with features
    test_df_ml = test_df.withColumn('weekofyear', weekofyear('date'))
    test_df_ml = test_df_ml.withColumn('month', month('date'))
    test_df_ml = test_df_ml.withColumn('dayofweek', dayofweek('date'))

    # # Add external signals
    # test_df_ml = test_df_ml.withColumn(
    #     'promo_flag', when(rand() < 0.1, 1).otherwise(0)
    # )
    # test_df_ml = test_df_ml.withColumn(
    #     'holiday_flag', when(col('month').isin([11, 12]), 1).otherwise(0)
    # )

    # For test data, we'll need to calculate lagged features differently
    # (In practice, you'd use the last values from training data)
    test_df_ml = test_df_ml.withColumn('lag_demand', col('demand'))
    test_df_ml = test_df_ml.withColumn('lag_demand_2', col('demand'))
    test_df_ml = test_df_ml.withColumn('lag_demand_3', col('demand'))
    test_df_ml = test_df_ml.withColumn('demand_ma_4', col('demand'))

    return test_df_ml

def add_column_with_logic(df, column_name, logic_func):
    return df.withColumn(column_name, logic_func())

def get_context_aware_forecast(categories, test_df_ml, models):
    # Generate predictions for each category
    predictions_list = []

    for category in categories:
        # Filter data for this category
        category_test = test_df_ml.filter(col('category') == category)
        
        # Make predictions
        predictions = models[category].transform(category_test)
        predictions_list.append(predictions)

    # Combine all predictions
    context_aware_predictions = predictions_list[0]
    for pred in predictions_list[1:]:
        context_aware_predictions = context_aware_predictions.union(pred)

    # Rename prediction column to forecast_demand
    context_aware_predictions = context_aware_predictions.withColumnRenamed('prediction', 'forecast_demand')

    return context_aware_predictions

# Calculate context-aware inventory parameters
def generate_context_aware_plan(context_aware_predictions, product_params, buffer_dict):
    context_aware_plan = context_aware_predictions.join(
        product_params.select(
            'product_id', 'safety_stock', 'demand_std', 'lead_time_avg'
        ),
        'product_id'
    )

    # Create a buffer mapping UDF
    buffer_mapping = create_map(
        [lit(x) for k, v in buffer_dict.items() for x in (k, v)]
    )

    context_aware_plan = context_aware_plan.withColumn(
        'buffer_factor', buffer_mapping[col('category')]
    )

    # Adjust forecast with buffer
    context_aware_plan = context_aware_plan.withColumn(
        'adjusted_forecast', col('forecast_demand') * (1 + col('buffer_factor'))
    )

    # Calculate adaptive safety stock (simplified version)
    context_aware_plan = context_aware_plan.withColumn(
        'adaptive_safety_stock', 
        col('safety_stock') * when(col('lead_time') > col('lead_time_avg'), 1.5).otherwise(1.0)
    )

    # Calculate required inventory and order quantity
    context_aware_plan = context_aware_plan.withColumn(
        'required_inventory', 
        col('adjusted_forecast') * col('lead_time') + col('adaptive_safety_stock')
    )

    context_aware_plan = context_aware_plan.withColumn(
        'order_quantity', 
        when(col('required_inventory') > 0, col('required_inventory')).otherwise(0)
    )

    # Select relevant columns
    context_aware_plan = context_aware_plan.select(
        'date', 'product_id', 'category', 'forecast_demand', 'demand',
        'lead_time', 'adaptive_safety_stock', 'order_quantity', 'required_inventory'
    ).withColumnRenamed('demand', 'actual_demand').withColumnRenamed('adaptive_safety_stock', 'safety_stock')

    return context_aware_plan

# UDF for inventory simulation
@udf(returnType=StructType([
    StructField("ending_inventory", DoubleType(), True),
    StructField("fulfilled_demand", DoubleType(), True),
    StructField("stockout", DoubleType(), True),
    StructField("holding_cost", DoubleType(), True),
    StructField("stockout_cost", DoubleType(), True),
    StructField("order_cost", DoubleType(), True),
    StructField("total_cost", DoubleType(), True),
    StructField("service_level", DoubleType(), True)
]))

def simulate_inventory_operation(starting_inventory, actual_demand, order_quantity, 
                               unit_cost, stockout_multiplier):
    """Simulate inventory operation for a single period"""
    # Process demand
    fulfilled_demand = builtins.min(starting_inventory, actual_demand)
    stockout = builtins.max(0, actual_demand - fulfilled_demand)
    ending_inventory = builtins.max(0, starting_inventory - actual_demand)
    
    # Process order
    new_inventory = ending_inventory + order_quantity
    
    # Calculate costs
    holding_cost = ending_inventory * (unit_cost * 0.25 / 52)  # Weekly holding cost
    stockout_cost = stockout * unit_cost * stockout_multiplier
    order_cost = 50 if order_quantity > 0 else 0  # Fixed ordering cost
    total_cost = holding_cost + stockout_cost + order_cost
    
    # Calculate service level
    service_level = fulfilled_demand / actual_demand if actual_demand > 0 else 1.0
    
    return (ending_inventory, fulfilled_demand, stockout, holding_cost, 
            stockout_cost, order_cost, total_cost, service_level)
    
def simulate_context_aware_approach(context_aware_plan, product_info_df):
    # Simulate context-aware approach
    context_aware_simulation = context_aware_plan.join(
        product_info_df.select('product_id', 'unit_cost', 'stockout_cost_multiplier'),
        'product_id'
    )

    # Initialize starting inventory with safety stock
    context_aware_simulation = context_aware_simulation.withColumn(
        'starting_inventory', col('safety_stock')
    )

    # Apply simulation UDF
    context_aware_simulation = context_aware_simulation.withColumn(
        'sim_results',
        simulate_inventory_operation(
            'starting_inventory', 'actual_demand', 'order_quantity',
            'unit_cost', 'stockout_cost_multiplier'
        )
    )

    # Expand simulation results
    context_aware_simulation = context_aware_simulation.select(
        'date', 'product_id', 'category', 'actual_demand',
        'sim_results.*'
    )

    return context_aware_simulation


def simulate_traditional_approach(traditional_plan, product_info_df):
    # Simulate traditional approach
    traditional_simulation = traditional_plan.join(
        product_info_df.select('product_id', 'unit_cost', 'stockout_cost_multiplier'),
        'product_id'
    )

    # Initialize starting inventory with safety stock
    traditional_simulation = traditional_simulation.withColumn(
        'starting_inventory', col('safety_stock')
    )

    # Apply simulation UDF
    traditional_simulation = traditional_simulation.withColumn(
        'sim_results',
        simulate_inventory_operation(
            'starting_inventory', 'actual_demand', 'order_quantity',
            'unit_cost', 'stockout_cost_multiplier'
        )
    )

    # Expand simulation results
    traditional_simulation = traditional_simulation.select(
        'date', 'product_id', 'category', 'actual_demand',
        'sim_results.*'
    )

    return traditional_simulation

def get_performance_metrics(traditional_simulation, context_aware_simulation):
    # Calculate overall performance metrics
    traditional_metrics = traditional_simulation.agg(
        sum('holding_cost').alias('total_holding_cost'),
        sum('stockout_cost').alias('total_stockout_cost'),
        sum('order_cost').alias('total_order_cost'),
        sum('total_cost').alias('total_cost'),
        avg('service_level').alias('avg_service_level'),
        (sum(when(col('stockout') > 0, 1).otherwise(0)) / count('*')).alias('stockout_frequency'),
        avg('ending_inventory').alias('avg_inventory')
    ).collect()[0]

    context_aware_metrics = context_aware_simulation.agg(
        sum('holding_cost').alias('total_holding_cost'),
        sum('stockout_cost').alias('total_stockout_cost'),
        sum('order_cost').alias('total_order_cost'),
        sum('total_cost').alias('total_cost'),
        avg('service_level').alias('avg_service_level'),
        (sum(when(col('stockout') > 0, 1).otherwise(0)) / count('*')).alias('stockout_frequency'),
        avg('ending_inventory').alias('avg_inventory')
    ).collect()[0]

    return traditional_metrics, context_aware_metrics

def get_comparison(traditional_metrics, context_aware_metrics):
    
    # Create comparison DataFrame
    comparison_data = [
        ('Total Cost', traditional_metrics['total_cost'], context_aware_metrics['total_cost'],
        (1 - context_aware_metrics['total_cost'] / traditional_metrics['total_cost']) * 100),
        ('Service Level', traditional_metrics['avg_service_level'], context_aware_metrics['avg_service_level'],
        (context_aware_metrics['avg_service_level'] - traditional_metrics['avg_service_level']) * 100),
        ('Stockout Frequency', traditional_metrics['stockout_frequency'], context_aware_metrics['stockout_frequency'],
        (1 - context_aware_metrics['stockout_frequency'] / traditional_metrics['stockout_frequency']) * 100),
        ('Avg Inventory', traditional_metrics['avg_inventory'], context_aware_metrics['avg_inventory'],
        (1 - context_aware_metrics['avg_inventory'] / traditional_metrics['avg_inventory']) * 100)
    ]

    comparison_df = spark.createDataFrame(
        comparison_data,
        ['Metric', 'Traditional', 'Context_Aware', 'Improvement_Percent']
    )

    return comparison_df

def get_category_comparison(traditional_simulation, context_aware_simulation):
    # Calculate category-specific metrics
    traditional_category_metrics = traditional_simulation.groupBy('category').agg(
        sum('total_cost').alias('total_cost'),
        avg('service_level').alias('service_level'),
        (sum(when(col('stockout') > 0, 1).otherwise(0)) / count('*')).alias('stockout_frequency'),
        avg('ending_inventory').alias('avg_inventory')
    )

    context_aware_category_metrics = context_aware_simulation.groupBy('category').agg(
        sum('total_cost').alias('total_cost'),
        avg('service_level').alias('service_level'),
        (sum(when(col('stockout') > 0, 1).otherwise(0)) / count('*')).alias('stockout_frequency'),
        avg('ending_inventory').alias('avg_inventory')
    )

    # Join for comparison
    category_comparison = traditional_category_metrics.alias('t').join(
        context_aware_category_metrics.alias('c'),
        col('t.category') == col('c.category')
    ).select(
        col('t.category'),
        col('t.service_level').alias('Traditional_Service_Level'),
        col('c.service_level').alias('Context_Aware_Service_Level'),
        ((col('c.service_level') - col('t.service_level')) * 100).alias('Service_Level_Improvement'),
        col('t.total_cost').alias('Traditional_Cost'),
        col('c.total_cost').alias('Context_Aware_Cost'),
        ((1 - col('c.total_cost') / col('t.total_cost')) * 100).alias('Cost_Reduction')
    )

    return category_comparison

def get_forecast_accuracy(traditional_plan, context_aware_plan):
    # Calculate forecast accuracy metrics
    # Traditional forecast accuracy
    traditional_eval = traditional_plan.withColumn(
        'error', abs(col('forecast_demand') - col('actual_demand'))
    ).withColumn(
        'squared_error', pow(col('forecast_demand') - col('actual_demand'), 2)
    ).withColumn(
        'percentage_error', 
        when(col('actual_demand') != 0, 
            abs(col('forecast_demand') - col('actual_demand')) / col('actual_demand')
        ).otherwise(0)
    )

    traditional_accuracy = traditional_eval.agg(
        avg('error').alias('MAE'),
        sqrt(avg('squared_error')).alias('RMSE'),
        avg('percentage_error').alias('MAPE')
    ).collect()[0]

    # Context-aware forecast accuracy
    context_eval = context_aware_plan.withColumn(
        'error', abs(col('forecast_demand') - col('actual_demand'))
    ).withColumn(
        'squared_error', pow(col('forecast_demand') - col('actual_demand'), 2)
    ).withColumn(
        'percentage_error',
        when(col('actual_demand') != 0,
            abs(col('forecast_demand') - col('actual_demand')) / col('actual_demand')
        ).otherwise(0)
    )

    context_accuracy = context_eval.agg(
        avg('error').alias('MAE'),
        sqrt(avg('squared_error')).alias('RMSE'),
        avg('percentage_error').alias('MAPE')
    ).collect()[0]

    # Create accuracy comparison
    accuracy_data = [
        ('MAE', traditional_accuracy['MAE'], context_accuracy['MAE'],
        (1 - context_accuracy['MAE'] / traditional_accuracy['MAE']) * 100),
        ('RMSE', traditional_accuracy['RMSE'], context_accuracy['RMSE'],
        (1 - context_accuracy['RMSE'] / traditional_accuracy['RMSE']) * 100),
        ('MAPE', traditional_accuracy['MAPE'], context_accuracy['MAPE'],
        (1 - context_accuracy['MAPE'] / traditional_accuracy['MAPE']) * 100)
    ]

    accuracy_df = spark.createDataFrame(
        accuracy_data,
        ['Metric', 'Traditional', 'Context_Aware', 'Improvement_Percent']
    )

    return accuracy_df

def get_product_comparison(traditional_simulation, context_aware_simulation, product_info_df):

    # Top 10 products with highest cost savings
    product_costs_traditional = traditional_simulation.groupBy('product_id').agg(
        sum('total_cost').alias('traditional_cost')
    )

    product_costs_context = context_aware_simulation.groupBy('product_id').agg(
        sum('total_cost').alias('context_cost')
    )

    product_comparison = product_costs_traditional.join(
        product_costs_context, 'product_id'
    ).join(
        product_info_df.select('product_id', 'category'), 'product_id'
    ).withColumn(
        'cost_saving', col('traditional_cost') - col('context_cost')
    ).withColumn(
        'saving_percentage', 
        (col('traditional_cost') - col('context_cost')) / col('traditional_cost') * 100
    ).orderBy(desc('cost_saving')).limit(10)

    return product_comparison  

