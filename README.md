# Don-Aborah-Petroineos-Challenge

This repository contains my solution for the **Petroineos Summer 2025 Data Analysis Internship** coding challenge. The core deliverable is the implementation of the `PowerPlant` class, designed to efficiently process, clean, merge, and aggregate power plant volume data from multiple CSV files.

## Overview

The `PowerPlants` class provides:

- Loading new power plant volume data from CSV files with robust error handling and strict data validation.
- Merging and updating an existing database of power plant data stored in `database.csv`, with deduplication based on the most recent update time.
- Creating timestamped backups of the database before overwriting to prevent data loss.
- Optional file-locking mechanisms (Linux/Unix systems) during save operations to prevent concurrent write corruption.
- Aggregation methods to summarize data monthly and by country/technology.
- Ensuring data consistency, handling missing values, and optimizing data types for performance and memory efficiency.

## Repository Contents

- `power_plants.py` — Full implementation of the `PowerPlant` class, including data loading, saving, and aggregation methods.
- Sample CSV files (`gas_plants.csv`, `wind_plants.csv`, etc.) used during development and testing are **not included** in this repository for privacy and size reasons.
- This README file explaining the project, usage, and design considerations.

## How to Use

1. Clone the repository.
2. Place your input CSV files (e.g., `gas_plants.csv`, `wind_plants.csv`) in the project directory.
3. Use the `PowerPlant` class from `power_plants.py` to load, process, and save data as shown:

```python
from power_plants import PowerPlant

pp = PowerPlant()

# Load new data from a CSV file
new_data = pp.load_new_data_from_file('wind_plants.csv')

# Save the loaded data into the database, merging with existing data
pp.save_new_data(new_data)
