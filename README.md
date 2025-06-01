# Don-Aborah-Petroineos-Challenge

This repository contains my solution for the **Petroineos Summer 2025 Data Analysis Internship** coding challenge. The core deliverable is the implementation of the `PowerPlants` class, designed to efficiently process, clean, and aggregate power plant volume data from multiple CSV files.

## Overview

The `PowerPlants` class supports:

- Loading new power plant volume data from CSV files with robust error handling and data validation.
- Merging and updating an existing database of power plant data stored in `database.csv`.
- Aggregating data to monthly and country-level summaries (to be implemented).
- Ensuring data consistency, handling missing values, and optimizing data types for performance.

## Repository Contents

- `power_plants.py` — Contains the `PowerPlants` class implementation with methods for data loading, saving, and aggregation.
- CSV files for testing (not included in public repo for privacy or size reasons).
- This README file describing the project.

## How to Use

1. Clone the repository.
2. Place your input CSV files (e.g., `gas_plants.csv`, `wind_plants.csv`, etc.) in the project directory.
3. Use the `PowerPlants` class in `power_plants.py` to load and process data:

```python
from power_plants import PowerPlants
pp = PowerPlants()
new_data = pp.load_new_data_from_file('wind_plants.csv')
pp.save_new_data(new_data)
