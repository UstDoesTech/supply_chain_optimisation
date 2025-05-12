# 📦 Context-Aware Inventory Optimization

This project simulates and compares two inventory optimization strategies—**traditional** and **context-aware**—across a multi-category supply chain. It includes:

- Forecasting with and without external context
- Dynamic buffer and reorder logic
- Simulation of real-world events (seasonality, volatility, disruptions)
- Performance comparison and Databricks AI/BI dashboard
- Ready-to-run Databricks notebook

---

## 🚀 Project Highlights

| Metric              | Traditional | Context-Aware | Improvement |
|---------------------|-------------|----------------|-------------|
| Stockout Frequency  | 0.54        | 0.21           | **+60.7%**  |
| Service Level       | 85%         | 96%            | **+10.8%**  |
| Total Cost (USD)    | $4.19M      | $1.62M         | **+61.3%**  |
| Forecast MAPE       | 0.22        | 0.10           | **+55.1%**  |

📉 **Result**: Context-aware planning dramatically improves outcomes—**even when holding more inventory.**

---

## 🧠 What's Inside

A Databricks Asset Bundle that will deploy the Notebook, wheel, and AI/BI Dashboard.

Just change the values in the `databricks.yml` file to match your environment.

## 🗨️ Genie Integration (Optional)
Use your Genie room to:

- Ask questions about simulation results
- Explore forecast and cost patterns by product or category
- Compare performance across strategies interactively

## 🙏 Acknowledgements

Inspired by real-world needs to adapt supply chains beyond rigid heuristics. Built with flexibility for experimentation and extensibility.
