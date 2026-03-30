# Factor-Based Portfolio Stress Testing & Risk Analytics Engine

A Python-based portfolio risk analysis platform that simulates macroeconomic stress scenarios, evaluates portfolio resilience, and supports both **single-portfolio deep analysis** and **multi-portfolio comparison** using a factor-driven framework.

---

## Overview

This project models how portfolios behave under different macroeconomic environments by applying **factor-based shocks** (equity, interest rates, inflation, commodities) to asset-level exposures.

It is designed to replicate how asset managers and risk teams:

* Analyze individual portfolios in depth
* Compare multiple strategies
* Make allocation decisions under uncertainty

---

## Core Capabilities

### Single Portfolio Analysis (Deep Dive Mode)

Designed for detailed portfolio diagnostics.

* Dynamic scenario stress testing
* Rolling backtesting vs realized performance
* Deep dive into portfolio behavior under stress
* Identification of key risk drivers

---

### Multi-Portfolio Comparison (Decision Mode)

Designed for portfolio selection and allocation.

* Scenario-based comparison across portfolios
* Relative performance under macro shocks
* Ranking using Resilience Score
* Identification of most resilient and highest-risk portfolios

---

## Key Features

### Factor-Based Stress Testing

* Models asset sensitivity to:

  * Equity markets
  * Interest rates
  * Inflation
  * Commodities
* Applies shocks at the **factor level**, enabling differentiated asset behavior

---

### Scenario Analysis

Predefined macro scenarios include:

* Equity Market Crash
* Yield Curve Shock
* Inflation Spike
* Stagflation

Each scenario is mapped to factor movements and propagated through asset-level sensitivities.

---

### Rolling Backtesting (Single Portfolio)

* Compares model outputs to historical outcomes
* Evaluates model accuracy and behavior over time
* Provides additional validation layer for stress scenarios

---

### Resilience Scoring System

* Ranks portfolios using a weighted scoring model based on:

  * Worst-case return
  * Maximum drawdown
  * Value at Risk (VaR)
* Produces a **Resilience Score (0–100)**
* Higher score = more resilient portfolio under stress

---

### Robust Data Ingestion Pipeline

* Handles messy real-world CSV inputs
* Automatically:

  * Detects column names (Ticker, Weight, etc.)
  * Cleans formatting inconsistencies
  * Normalizes weights
* Ensures reliable processing without strict input requirements

---

## How It Works

1. Upload one or more portfolio CSV files
2. System automatically determines execution mode:

   * 1 portfolio → Single Analysis Mode
   * Multiple portfolios → Comparison Mode
3. Assets are mapped to factor sensitivities (betas)
4. Scenarios are translated into factor shocks
5. Asset-level returns are computed
6. Portfolio results are aggregated and visualized
7. Portfolios are ranked (in multi mode) using Resilience Score

---

## Example Use Cases

* Analyze a single portfolio’s sensitivity to macro shocks
* Backtest portfolio behavior against historical data
* Compare growth vs defensive strategies
* Identify the most resilient allocation under stress
* Support investment decision-making and portfolio construction

---

## Tech Stack

* Python
* Streamlit
* Pandas / NumPy
* Plotly

---

## Input Format

Supports flexible CSV formats, including:

* Ticker / Symbol / Asset
* Weight / Allocation / %

Example:

Ticker,Weight
AAPL,0.25
MSFT,0.25

---

## Future Improvements & Roadmap

### Data & Model Enhancements

* Regression-based beta estimation
* Historical scenario calibration (e.g., 2008, COVID)

---

### Risk Modeling & Analytics

* Enhanced Resilience Score (downside penalties, full distribution)
* Monte Carlo simulation for tail risk

---

### Portfolio Intelligence Features

* Factor exposure dashboard
* Sector & asset classification engine

---

### User Interaction & Flexibility

* Custom scenario builder
* Portfolio optimization module

---

### Platform-Level Enhancements

* Live data integration
* Portfolio tracking & dashboarding

---

## Goal

To build a **practical portfolio risk and decision engine** that combines macroeconomic intuition with quantitative modeling to support real-world investment decisions.

---

## Author

Ethan Wong
Finance, Accounting & Economics student @Purdue Univeristy | Interested in Markets and Tech


