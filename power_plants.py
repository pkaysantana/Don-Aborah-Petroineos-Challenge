import pandas as pd
import logging
import os # Import the os module for file path operations
import datetime # Import datetime for timestamped backups
import fcntl # For file locking (Linux/Unix-specific) - mention it's illustrative and might be overkill

# Configure logger (at the module level or appropriate scope)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PowerPlant(object):
    def __init__(self):
        self.database_file = 'database.csv'
    
    def load_new_data_from_file(self, file_path: str, country_mapping: dict = None) -> pd.DataFrame:
        """
        Loads power plant volume data from a CSV file, processes it, and prepares it for database saving.

        This method performs the following operations:
        - Reads the CSV file into a pandas DataFrame, handling file-related errors.
        - Standardises column names (strips whitespace, converts to lowercase).
        - Validates the presence of required columns ('symbol', 'updatetime', 'volume', 'country', 'technology').
        - Fills missing 'volume' data with zero.
        - Converts country codes to full country names using a provided or default mapping.
        - Parses 'updatetime' column to datetime objects and drops rows where conversion fails.
        - Removes duplicate rows to ensure data consistency.
        - Optimises data types for better memory usage and performance where possible.

        Args:
            file_path (str): The absolute or relative path to the CSV file containing power plant data.
            country_mapping (dict, optional): A dictionary to map country codes (keys) to full names (values).
                                             If None, a default mapping will be used.

        Returns:
            pd.DataFrame: A processed pandas DataFrame ready for saving into the database.
                          Returns an empty DataFrame if the file is empty.

        Raises:
            FileNotFoundError: If the specified `file_path` does not exist.
            IOError: For other general issues encountered during file reading (e.g., permission denied).
            ValueError: If critical required columns are missing from the input CSV file.
        """
        # Previous implementation commented out:
        # df = pd.read_csv(file_path)
        # df.fillna(0, inplace=True)

        # # Implement country code conversion
        # country_map = {
        #     'FR': 'France',
        #     'DE': 'Germany',
        #     'ES': 'Spain',
        #     'GB': 'United Kingdom',
        #     'IT': 'Italy'
        # }
        # if 'country' in df.columns:
        #     df['country'] = df['country'].map(country_map).fillna(df['country'])

        # # Ensure 'updatetime' column is in datetime format
        # if 'updatetime' in df.columns:
        #     df['updatetime'] = pd.to_datetime(df['updatetime'], errors='coerce') # 'coerce' will turn unparseable dates into NaT

        # Previous implementation commented out:
        # df = pd.read_csv(file_path)
        
        # # Standardise column names
        # df.columns = [col.strip().lower() for col in df.columns]
        
        # # Validate required columns
        # required_cols = ['symbol', 'updatetime', 'volume', 'country', 'technology']
        # missing_cols = [col for col in required_cols if col not in df.columns]
        # if missing_cols:
        #     raise ValueError(f"Missing required columns: {missing_cols}")
        
        # # Fill missing volume data with 0
        # df['volume'] = df['volume'].fillna(0)
        
        # # Convert country codes to full names
        # country_map = {'fr': 'France', 'de': 'Germany', 'uk': 'United Kingdom', 'es': 'Spain', 'it': 'Italy'} # Added ES and IT to map
        # df['country'] = df['country'].str.lower().map(country_map).fillna(df['country'])
        
        # # Parse updatetime column as datetime
        # df['updatetime'] = pd.to_datetime(df['updatetime'], errors='coerce')
        # df = df.dropna(subset=['updatetime'])
        
        # print(f"Loaded {len(df)} records from {file_path}")
        
        # return df

        # Previous implementation:

        # Define a default country mapping if none is provided
        default_country_map = {
            'fr': 'France', 'de': 'Germany', 'uk': 'United Kingdom',
            'es': 'Spain', 'it': 'Italy', 'ch': 'Switzerland',
            'at': 'Austria', 'be': 'Belgium', 'nl': 'Netherlands',
            'se': 'Sweden', 'no': 'Norway', 'fi': 'Finland'
        }
        country_map_to_use = country_mapping if country_mapping is not None else default_country_map

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully read CSV file: '{file_path}'. Initial records: {len(df)}")
            
            if df.empty:
                logger.warning(f"The file '{file_path}' is empty. Returning an empty DataFrame.")
                return pd.DataFrame()

        except FileNotFoundError:
            logger.error(f"Error: The file '{file_path}' was not found. Please check the path.")
            raise
        except pd.errors.EmptyDataError:
            logger.warning(f"Warning: The file '{file_path}' is empty. Returning an empty DataFrame.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"An unexpected error occurred while reading '{file_path}': {e}")
            raise IOError(f"Failed to read file {file_path}: {e}")

        # Standardise column names (strip whitespace and convert to lowercase)
        # Validate required columns upfront to avoid errors in subsequent chained operations
        df.columns = [col.strip().lower() for col in df.columns]
        required_cols = ['symbol', 'updatetime', 'volume', 'country', 'technology']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns in '{file_path}': {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Chain pandas operations for cleaner code:
        df = (
            df
            .assign(
                # Fill missing 'volume' data with 0
                volume=df['volume'].fillna(0),
                # Convert country codes to full names, apply lowercase to country column before mapping
                country=df['country'].str.lower().map(country_map_to_use).fillna(df['country']),
                # Parse 'updatetime' column to datetime, coercing errors to NaT
                updatetime=pd.to_datetime(df['updatetime'], errors='coerce')
            )
            # Drop rows where 'updatetime' could not be parsed (resulted in NaT)
            .dropna(subset=['updatetime'])
            # Remove duplicate rows based on all columns to ensure data consistency
            .drop_duplicates()
        )

        # Data type optimisation (after main processing)
        # Convert columns to more memory-efficient types where appropriate
        optimal_dtypes = {
            'symbol': 'category',
            'volume': 'float32', # 'float32' might be sufficient for volume data
            'country': 'category',
            'technology': 'category'
        }
        for col, dtype in optimal_dtypes.items():
            if col in df.columns and df[col].dtype != dtype:
                try:
                    df[col] = df[col].astype(dtype)
                except TypeError as e:
                    logger.warning(f"Could not convert column '{col}' to '{dtype}' in '{file_path}': {e}")
                except Exception as e:
                    logger.warning(f"An unexpected error occurred during dtype conversion for '{col}' in '{file_path}': {e}")


        logger.info(f"Processed and loaded {len(df)} valid records from '{file_path}'.")
        return df

    def _load_database(self) -> pd.DataFrame:
        """Helper method to load existing data from the database file."""
        db_path_abs = os.path.abspath(self.database_file)
        try:
            if os.path.exists(self.database_file):
                existing_data = pd.read_csv(self.database_file)
                logger.debug(f"Initial load of existing database from '{db_path_abs}'. Records: {len(existing_data)}")
                # Ensure updatetime is datetime type for existing data
                if 'updatetime' in existing_data.columns:
                    initial_nat_count = existing_data['updatetime'].isna().sum()
                    existing_data['updatetime'] = pd.to_datetime(existing_data['updatetime'], errors='coerce')
                    nats_after_conversion = existing_data['updatetime'].isna().sum()
                    if nats_after_conversion > initial_nat_count:
                        logger.warning(f"Dropped {nats_after_conversion - initial_nat_count} rows from existing database due to unparseable 'updatetime' values in '{db_path_abs}'.")
                    existing_data = existing_data.dropna(subset=['updatetime']) # Drop rows where conversion failed
                
                # Validate data types for critical columns in existing data
                if 'symbol' in existing_data.columns:
                    existing_data['symbol'] = existing_data['symbol'].astype('category')
                
                logger.info(f"Loaded {len(existing_data)} records from existing database: '{db_path_abs}'.")
                return existing_data
            else:
                logger.info(f"Database file '{db_path_abs}' not found. Initializing an empty DataFrame.")
                return pd.DataFrame() # Create empty DataFrame if file doesn't exist
        except pd.errors.EmptyDataError:
            logger.warning(f"Database file '{db_path_abs}' is empty. Initializing an empty DataFrame.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading existing database from '{db_path_abs}': {e}")
            raise IOError(f"Failed to load database from '{self.database_file}': {e}")

    def _merge_and_deduplicate(self, existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """Helper method to combine data and remove duplicates based on most recent updatetime."""
        # Check column consistency before concatenation if existing data is not empty
        if not existing_data.empty and not new_data.empty:
            existing_cols = set(existing_data.columns)
            new_cols = set(new_data.columns)
            
            if existing_cols != new_cols:
                added_cols = list(new_cols - existing_cols)
                dropped_cols = list(existing_cols - new_cols)
                if added_cols:
                    logger.warning(f"New columns added in input data: {added_cols}. Schema evolution detected.")
                if dropped_cols:
                    logger.warning(f"Columns missing in input data that exist in database: {dropped_cols}. Schema evolution detected.")
                
                logger.debug(f"Column mismatch between existing data {existing_cols} and new data {new_cols}. Attempting to align columns.")
                # Align columns by reindexing new_data to match existing_data columns, filling missing with NaN
                # and adding new columns from new_data if they don't exist in existing_data
                all_cols = list(existing_cols.union(new_cols))
                existing_data = existing_data.reindex(columns=all_cols)
                new_data = new_data.reindex(columns=all_cols)
                logger.debug(f"Columns after alignment: {all_cols}")
        
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        logger.debug(f"Combined data has {len(combined_data)} records before de-duplication.")

        # Ensure 'updatetime' is datetime type for sorting in the combined data
        if 'updatetime' in combined_data.columns:
            # This check is already done in _load_database and should be handled by load_new_data_from_file
            # but as a failsafe before sorting, we ensure it's datetime.
            initial_nat_count = combined_data['updatetime'].isna().sum()
            combined_data['updatetime'] = pd.to_datetime(combined_data['updatetime'], errors='coerce')
            nats_after_conversion = combined_data['updatetime'].isna().sum()
            if nats_after_conversion > initial_nat_count:
                logger.warning(f"Dropped {nats_after_conversion - initial_nat_count} rows from combined data due to unparseable 'updatetime' values during merging.")
            combined_data = combined_data.dropna(subset=['updatetime']) # Drop rows where conversion failed


        # Sort by 'symbol' and 'updatetime' (descending) to ensure the most recent record for each symbol comes first
        # Only sort if 'symbol' and 'updatetime' columns exist, otherwise drop duplicates might not work as expected
        if 'symbol' in combined_data.columns and 'updatetime' in combined_data.columns:
            combined_data = combined_data.sort_values(by=['symbol', 'updatetime'], ascending=[True, False])
            final_data = combined_data.drop_duplicates(subset=['symbol'], keep='first')
        else:
            logger.warning("Missing 'symbol' or 'updatetime' in combined data. Skipping de-duplication by time. Keeping all unique rows.")
            final_data = combined_data.drop_duplicates() # Fallback to dropping all duplicates

        logger.debug(f"Final data has {len(final_data)} unique records after de-duplication.")
        return final_data

    def _save_database(self, data: pd.DataFrame):
        """Helper method to save the DataFrame to the database file."""
        db_path_abs = os.path.abspath(self.database_file)
        # Create a timestamped backup before overwriting
        if os.path.exists(self.database_file):
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            backup_file = f"{self.database_file}.backup_{timestamp}"
            try:
                os.replace(self.database_file, backup_file) # Atomically rename for backup
                logger.info(f"Created a timestamped backup of '{self.database_file}' at '{backup_file}'.")
            except OSError as e:
                logger.error(f"Error creating backup of '{self.database_file}': {e}")
                # Continue without backup if it fails, but log the error.

        try:
            # File locking (Illustrative, might be overkill for simple CSVs and single processes)
            # This is primarily for concurrent access in multi-process scenarios on Unix-like systems.
            # For Windows, alternative mechanisms like `msvcrt.locking` or higher-level database systems are needed.
            if os.name == 'posix': # Only attempt file lock on POSIX systems
                with open(self.database_file, 'a') as f: # Open in append mode to get a file descriptor
                    fcntl.flock(f, fcntl.LOCK_EX) # Acquire an exclusive lock
                    try:
                        data.to_csv(self.database_file, index=False)
                        logger.info(f"Successfully saved {len(data)} records to '{db_path_abs}'.")
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN) # Release the lock
            else:
                data.to_csv(self.database_file, index=False)
                logger.info(f"Successfully saved {len(data)} records to '{db_path_abs}'.")

        except Exception as e:
            logger.error(f"Error saving data to '{db_path_abs}': {e}")
            raise IOError(f"Failed to save data to database: {e}")

    def save_new_data(self, input_data: pd.DataFrame):
        """
        Saves new data to the database by merging with existing records and updating based on the most recent 'updatetime'.

        This method performs the following operations:
        - Loads the existing database from 'database.csv'. If the file does not exist, it initialises an empty DataFrame.
        - Validates the input data for required columns and critical nulls.
        - Merges the `input_data` DataFrame with the existing database.
        - Handles potential duplicate entries by keeping only the most recent record for each 'symbol'
          based on the 'updatetime' column.
        - Creates a timestamped backup of the existing database before saving.
        - Saves the consolidated and updated DataFrame back to 'database.csv', overwriting the previous content.
        - (Optional: Incorporates file locking to prevent data corruption from concurrent access, if applicable to OS.)

        Args:
            input_data (pd.DataFrame): A pandas DataFrame containing new power plant volume data to be saved.
                                       It should have 'symbol' and 'updatetime' columns for merging and updating.

        Raises:
            IOError: If there are issues reading from or writing to 'database.csv'.
            ValueError: If 'symbol' or 'updatetime' columns are missing from the input_data,
                        or if critical columns in input_data contain unexpected null values.
        """
        # Previous implementation commented out:
        # if 'symbol' not in input_data.columns or 'updatetime' not in input_data.columns:
        #     logger.error("Input data must contain 'symbol' and 'updatetime' columns for merging.")
        #     raise ValueError("Missing required columns: 'symbol' or 'updatetime' in input_data.")

        # try:
        #     # Load existing data from the database file
        #     if pd.io.common.file_exists(self.database_file) and pd.read_csv(self.database_file).empty:
        #         existing_data = pd.DataFrame(columns=input_data.columns) # Ensure consistent columns if file is empty but exists
        #         logger.info(f"Existing database '{self.database_file}' found but is empty. Initialising with input data columns.")
        #     elif pd.io.common.file_exists(self.database_file):
        #         existing_data = pd.read_csv(self.database_file)
        #         # Ensure updatetime is datetime type for existing data too
        #         if 'updatetime' in existing_data.columns:
        #             existing_data['updatetime'] = pd.to_datetime(existing_data['updatetime'], errors='coerce')
        #         logger.info(f"Loaded {len(existing_data)} records from existing database '{self.database_file}'.")
        #     else:
        #         existing_data = pd.DataFrame() # Create empty DataFrame if file doesn't exist
        #         logger.info(f"Database file '{self.database_file}' not found. A new one will be created.")
        # except pd.errors.EmptyDataError:
        #     existing_data = pd.DataFrame()
        #     logger.warning(f"Database file '{self.database_file}' is empty. Initialising an empty DataFrame.")
        # except Exception as e:
        #     logger.error(f"Error loading existing database from '{self.database_file}': {e}")
        #     raise IOError(f"Failed to load database: {e}")

        # # Combine existing and new data
        # combined_data = pd.concat([existing_data, input_data], ignore_index=True)
        # logger.info(f"Combined data has {len(combined_data)} records before de-duplication.")

        # # Sort by 'symbol' and 'updatetime' (descending) to ensure the most recent record for each symbol comes first
        # combined_data = combined_data.sort_values(by=['symbol', 'updatetime'], ascending=[True, False])

        # # Drop duplicates based on 'symbol', keeping the first occurrence (which is the most recent)
        # final_data = combined_data.drop_duplicates(subset=['symbol'], keep='first')
        # logger.info(f"Final data has {len(final_data)} unique records after de-duplication based on most recent 'updatetime'.")

        # try:
        #     # Save the updated DataFrame back to the database file
        #     final_data.to_csv(self.database_file, index=False)
        #     logger.info(f"Successfully saved {len(final_data)} records to '{self.database_file}'.")
        # except Exception as e:
        #     logger.error(f"Error saving data to '{self.database_file}': {e}")
        #     raise IOError(f"Failed to save data to database: {e}")

        # New and improved implementation:

        # 1. Handle empty input gracefully
        if input_data.empty:
            logger.info("Input data is empty. Skipping save operation.")
            return

        # 2. Validate required columns in input_data
        required_cols = ['symbol', 'updatetime']
        missing_cols = [col for col in required_cols if col not in input_data.columns]
        if missing_cols:
            logger.error(f"Input data must contain all required columns for merging. Missing: {missing_cols}")
            raise ValueError(f"Missing required columns in input_data: {missing_cols}")

        # 3. Validate data types for critical columns and enforce no unexpected nulls in keys
        if not pd.api.types.is_string_dtype(input_data['symbol']) and not pd.api.types.is_categorical_dtype(input_data['symbol']):
            logger.warning("Converting 'symbol' column in input data to category type.")
            input_data['symbol'] = input_data['symbol'].astype('category')
        
        if input_data['symbol'].isnull().any():
            logger.error("Critical null values found in 'symbol' column of input data. Cannot proceed with merging.")
            raise ValueError("Null values found in 'symbol' column of input_data, which is a critical key.")

        # Ensure 'updatetime' in input_data is datetime type
        if 'updatetime' in input_data.columns:
            input_data_initial_nat_count = input_data['updatetime'].isna().sum()
            input_data['updatetime'] = pd.to_datetime(input_data['updatetime'], errors='coerce')
            input_data_nats_after_conversion = input_data['updatetime'].isna().sum()
            if input_data_nats_after_conversion > input_data_initial_nat_count:
                logger.warning(f"Dropped {input_data_nats_after_conversion - input_data_initial_nat_count} rows from input data due to unparseable 'updatetime' values before saving.")
            input_data = input_data.dropna(subset=['updatetime']) # Drop rows where conversion failed
            if input_data['updatetime'].isnull().any():
                logger.error("Critical null values found in 'updatetime' column of input data after conversion. Cannot proceed with merging.")
                raise ValueError("Null values found in 'updatetime' column of input_data after processing, which is a critical key.")


        # 4. Load existing data using helper method
        existing_data = self._load_database()

        # 5. Merge and deduplicate using helper method
        final_data = self._merge_and_deduplicate(existing_data, input_data)

        # 6. Save the final data using helper method (includes backup and locking)
        self._save_database(final_data)

    def get_data_from_database(self):
        """
        Retrieve data from the database.
        """
        # Implementation to read data from the database
        pass

    def aggregate_data_to_monthly(self):
        """
        Aggregate data to monthly level.
        """
        # Implementation to aggregate data to monthly level
        pass

    def aggregate_data_to_country(self):
        """
        Aggregate data to country level.
        """
        # Implementation to aggregate data to country level
        pass