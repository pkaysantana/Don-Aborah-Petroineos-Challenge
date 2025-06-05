# Don-Aborah-Petroineos-Challenge

This repository contains my solution for the **Petroineos Summer 2025 Data Analysis Internship** coding challenge. The core deliverable is the implementation of the `PowerPlants` class, designed to efficiently process, clean, merge, and aggregate power plant volume data from multiple CSV files.

## Overview

The `PowerPlants` class provides:

- Loading new power plant volume data from CSV files with robust error handling and strict data validation.
- Merging and updating an existing database of power plant data stored in `database.csv`, with deduplication based on the most recent update time.
- Creating timestamped backups of the database before overwriting to prevent data loss.
- Optional file-locking mechanisms (Linux/Unix systems) during save operations to prevent concurrent write corruption.
- Aggregation methods to summarize data monthly and by country/technology.
- Ensuring data consistency, handling missing values, and optimizing data types for performance and memory efficiency.

## Repository Contents

- `power_plants.py` — Full implementation of the `PowerPlants` class, including data loading, saving, and aggregation methods.
- Sample CSV files (`gas_plants.csv`, `wind_plants.csv`, etc.) used during development and testing are **not included** in this repository for privacy and size reasons.
- This README file explaining the project, usage, and design considerations.

## How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/pkaysantana/Don-Aborah-Petroineos-Challenge.git
   cd Don-Aborah-Petroineos-Challenge
