from __future__ import annotations
import pandas as pd
import logging
import os # Import the os module for file path operations
import datetime # Import datetime for timestamped backups
# import fcntl # For file locking (Linux/Unix-specific) -  it's illustrative and might be overkill
# from portalocker import Lock, exceptions as portalocker_exceptions # Consider for cross-platform locking

# Configure logger (at the module level or appropriate scope)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PowerPlants(object):
    # Class-level constants for default aggregation parameters
    _DEFAULT_MONTHLY_GROUPBY_COLS = ['symbol', 'technology', 'country']
    _DEFAULT_MONTHLY_AGG_FUNCTIONS = {'average_volume': 'mean', 'min_volume': 'min', 'max_volume': 'max'}
    _DEFAULT_COUNTRY_GROUPBY_COLS = ['country', 'technology']
    _DEFAULT_COUNTRY_AGG_FUNCTIONS = {'total_volume': 'sum'}

    def __init__(self):
        self.database_file = 'database.csv'
        self.max_backups = 5 # Configuration for maximum number of backups
        self._cached_database_data = None  # Cache for the full raw database data
        self._cached_monthly_data = None   # Cache for monthly aggregated data
        self._cached_country_data = None   # Cache for country aggregated data
        self._last_db_modification_time = None # Timestamp of last database modification

    def load_new_data_from_file(self, file_path: str, country_mapping: dict = None) -> pd.DataFrame:
        """
        Loads power plant volume data from a CSV file, processes it, and prepares it for database saving.

        This method performs the following operations:
        - Reads the CSV file into a pandas DataFrame, handling file-related errors.
        - Renames specified columns and ensures all column names are lowercase and stripped of whitespace.
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
        # Define a default country mapping if none is provided
        default_country_map = {
            'fr': 'France', 'de': 'Germany', 'uk': 'United Kingdom',
            'es': 'Spain', 'it': 'Italy', 'ch': 'Switzerland',
            'at': 'Austria', 'be': 'Belgium', 'nl': 'Netherlands',
            'se': 'Sweden', 'no': 'Norway', 'fi': 'Finland'
        }
        country_map_to_use = country_mapping if country_mapping is not None else default_country_map

        # Initialize df as an empty DataFrame to ensure it's always defined
        df = pd.DataFrame() 

        try:
            # Attempt to read a preview of the file to detect headers
            # Use encoding='utf-8-sig' to handle BOM character during preview
            df_preview = pd.read_csv(file_path, nrows=5, encoding='utf-8-sig')
            logger.debug(f"Preview of first 5 rows from '{file_path}':\n{df_preview.head()}")

            # Clean preview column names for robust comparison (strip whitespace and quotes)
            preview_columns_cleaned = df_preview.columns.str.strip().str.strip("'\"").tolist()

            # Define expected raw headers for detection (using original casing)
            expected_raw_headers = ['Date', 'Country', 'Technology', 'SiteName', 'Volume']
            
            # Check if expected headers are present in the first row (default header=0)
            # We check against the cleaned preview columns (case-insensitive check)
            if all(header.lower() in [col.lower() for col in preview_columns_cleaned] for header in expected_raw_headers):
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                logger.debug(f"Headers detected on row 0. Loaded {len(df)} records from '{file_path}'.")
            else:
                try:
                    df = pd.read_csv(file_path, header=1, encoding='utf-8-sig')
                    logger.debug(f"Headers not detected on row 0. Attempting to load with header on row 1. Loaded {len(df)} records from '{file_path}'.")
                except Exception as inner_e:
                    logger.warning(f"Could not definitively detect headers on row 1 ({inner_e}). Falling back to default header=0 and relying on subsequent column validation.")
                    df = pd.read_csv(file_path, encoding='utf-8-sig')

        except FileNotFoundError:
            logger.error(f"Error: The file '{file_path}' was not found. Returning an empty DataFrame.")
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            logger.warning(f"Warning: The file '{file_path}' is empty (EmptyDataError). Returning an empty DataFrame.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"An unexpected error occurred while reading '{file_path}': {e}. Returning an empty DataFrame.")
            return pd.DataFrame()

        # If df is empty after reading (e.g., empty file or no valid data rows after header detection), return early.
        if df.empty:
            logger.warning(f"DataFrame is empty after initial file read and header detection. Returning empty DataFrame.")
            return pd.DataFrame()

        # Debug print immediately after reading the CSV
        print("Columns after read:", df.columns.tolist())

        # Clean column headers: strip whitespace and any literal quotes
        df.columns = df.columns.str.strip().str.strip("'\"")
        print("Columns after cleaning:", df.columns.tolist())

        # Rename columns after cleaning, mapping original casing to internal names
        rename_map = {
            'Date': 'updatetime',
            'Volume': 'volume',
            'Country': 'country',
            'Technology': 'technology',
            'SiteName': 'symbol'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Convert all column names to lowercase after renaming for internal consistency
        df.columns = df.columns.str.lower()
        print("Columns after renaming:", df.columns.tolist())

        # 4. Verify that required columns now exist exactly as expected:
        required_cols = ['symbol', 'updatetime', 'volume', 'country', 'technology']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns after processing: {missing_cols}")
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
        """
        Helper method to load existing data from the database file.

        Returns:
            pd.DataFrame: The DataFrame loaded from the database file. Returns an empty DataFrame
                          if the file does not exist or is empty.

        Raises:
            IOError: If there are issues reading the database file.
        """
        db_path_abs = os.path.abspath(self.database_file)
        try:
            if os.path.exists(self.database_file):
                existing_data = pd.read_csv(self.database_file)
                logger.debug(f"Initial load of existing database from '{db_path_abs}'. Records: {len(existing_data)}")
                
                # Normalise string columns and validate data types in existing data
                for col in ['symbol', 'country', 'technology']:
                    if col in existing_data.columns:
                        # Replace deprecated pd.api.types.is_string_dtype with pandas.api.types.is_string_dtype
                        if pd.api.types.is_string_dtype(existing_data[col]):
                            existing_data[col] = existing_data[col].str.strip().str.lower()
                        # Use isinstance with pd.CategoricalDtype instead of deprecated is_categorical_dtype
                        # This future-proofs the code against deprecation warnings.
                        if not isinstance(existing_data[col].dtype, pd.CategoricalDtype):
                            existing_data[col] = existing_data[col].astype('category')

                # Ensure updatetime is datetime type for existing data
                if 'updatetime' in existing_data.columns:
                    initial_nat_count = existing_data['updatetime'].isna().sum()
                    existing_data['updatetime'] = pd.to_datetime(existing_data['updatetime'], errors='coerce')
                    nats_after_conversion = existing_data['updatetime'].isna().sum()
                    if nats_after_conversion > initial_nat_count:
                        logger.warning(f"Dropped {nats_after_conversion - initial_nat_count} rows from existing database due to unparseable 'updatetime' values in '{db_path_abs}'.")
                    existing_data = existing_data.dropna(subset=['updatetime']) # Drop rows where conversion failed
                
                logger.info(f"Loaded {len(existing_data)} records from existing database: '{db_path_abs}'.")
                return existing_data
            else:
                logger.info(f"Database file '{db_path_abs}' not found. Initialising an empty DataFrame.")
                return pd.DataFrame() # Create empty DataFrame if file doesn't exist
        except pd.errors.EmptyDataError:
            logger.warning(f"Database file '{db_path_abs}' is empty. Initialising an empty DataFrame.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading existing database from '{db_path_abs}': {e}")
            raise IOError(f"Failed to load database from '{self.database_file}': {e}")

    def _merge_and_deduplicate(self, existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method to combine data and remove duplicates based on most recent updatetime.

        Args:
            existing_data (pd.DataFrame): The DataFrame containing existing records.
            new_data (pd.DataFrame): The DataFrame containing new records to merge.

        Returns:
            pd.DataFrame: A combined and deduplicated DataFrame with the most recent records.
        """
        # Normalise string columns in new_data before merging
        for col in ['symbol', 'country', 'technology']:
            if col in new_data.columns:
                # Replace deprecated pd.api.types.is_string_dtype with pandas.api.types.is_string_dtype
                if pd.api.types.is_string_dtype(new_data[col]):
                    new_data[col] = new_data[col].str.strip().str.lower()
                # Use isinstance with pd.CategoricalDtype instead of deprecated is_categorical_dtype
                # This future-proofs the code against deprecation warnings.
                if not isinstance(new_data[col].dtype, pd.CategoricalDtype): # Convert to category for memory optimisation if desired
                    new_data[col] = new_data[col].astype('category')
        
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
            # Re-confirming datetime type conversion as a final safeguard before sorting
            initial_nat_count = combined_data['updatetime'].isna().sum()
            combined_data['updatetime'] = pd.to_datetime(combined_data['updatetime'], errors='coerce')
            nats_after_conversion = combined_data['updatetime'].isna().sum()
            if nats_after_conversion > initial_nat_count:
                logger.warning(f"Dropped {nats_after_conversion - initial_nat_count} rows from combined data due to unparseable 'updatetime' values during merging.")
            combined_data = combined_data.dropna(subset=['updatetime']) # Drop rows where conversion failed
            if combined_data['updatetime'].isnull().any():
                logger.warning("Null values found in 'updatetime' column after processing in combined data. This might affect deduplication.")
                # This could be handled by dropping these rows or raising an error depending on desired behaviour
        
        # Validate that 'symbol' and 'updatetime' have no nulls before sorting/deduplication
        if 'symbol' in combined_data.columns and combined_data['symbol'].isnull().any():
            logger.warning("Null values detected in 'symbol' column after concatenation. These rows might be problematic for deduplication.")
            # Decide on strategy: drop rows with null 'symbol', or handle specifically. For now, they might be dropped by drop_duplicates if subset doesn't include 'symbol'
        
        if 'updatetime' in combined_data.columns and combined_data['updatetime'].isnull().any():
            logger.warning("Null values detected in 'updatetime' column after concatenation. These rows might not be correctly ordered for deduplication.")
            # Drop rows with NaT from updatetime if not already done, or handle as per policy.
            
        # Sort by 'symbol' and 'updatetime' (descending) to ensure the most recent record for each symbol comes first
        # Only sort if 'symbol' and 'updatetime' columns exist, otherwise drop duplicates might not work as expected
        if 'symbol' in combined_data.columns and 'updatetime' in combined_data.columns:
            # Ensure 'symbol' is sortable (e.g., not mixed types)
            combined_data['symbol'] = combined_data['symbol'].astype(str) # Convert to string for consistent sorting
            
            combined_data = combined_data.sort_values(by=['symbol', 'updatetime'], ascending=[True, False])
            final_data = combined_data.drop_duplicates(subset=['symbol'], keep='first')
        else:
            logger.warning("Missing 'symbol' or 'updatetime' in combined data (after alignment). Skipping de-duplication by time. Keeping all unique rows based on all columns.")
            final_data = combined_data.drop_duplicates() # Fallback to dropping all duplicates

        logger.info(f"Final data has {len(final_data)} unique records after de-duplication.")
        return final_data

    def _save_database(self, data: pd.DataFrame):
        """
        Helper method to save the DataFrame to the database file, with backup and optional locking.

        Args:
            data (pd.DataFrame): The DataFrame to be saved to the database file.

        Raises:
            IOError: If there are issues writing to the database file.
        """
        db_path_abs = os.path.abspath(self.database_file)
        
        # Manage backups: Keep only the last N backups
        backup_dir = os.path.dirname(self.database_file) or '.'
        backup_prefix = os.path.basename(self.database_file) + ".backup_"
        
        existing_backups = []
        for f_name in os.listdir(backup_dir):
            if f_name.startswith(backup_prefix) and os.path.isfile(os.path.join(backup_dir, f_name)):
                try:
                    # Extract timestamp from filename for sorting
                    timestamp_str = f_name[len(backup_prefix):]
                    timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                    existing_backups.append((timestamp, os.path.join(backup_dir, f_name)))
                except ValueError:
                    logger.debug(f"Could not parse timestamp from backup file: {f_name}")
                    continue
        
        existing_backups.sort() # Sort by timestamp (oldest first)
        
        # Delete oldest backups if exceeding limit
        if len(existing_backups) >= self.max_backups:
            num_to_delete = len(existing_backups) - self.max_backups + 1
            for i in range(num_to_delete):
                oldest_backup_path = existing_backups[i][1]
                try:
                    os.remove(oldest_backup_path)
                    logger.info(f"Deleted old backup: '{oldest_backup_path}'.")
                except OSError as e:
                    logger.error(f"Error deleting old backup '{oldest_backup_path}': {e}")
        
        # Create a timestamped backup before overwriting current database file
        if os.path.exists(self.database_file):
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            current_backup_file = f"{self.database_file}.backup_{timestamp}"
            try:
                os.replace(self.database_file, current_backup_file) # Atomically rename for backup
                logger.info(f"Created a timestamped backup of '{self.database_file}' at '{current_backup_file}'.")
            except OSError as e:
                logger.error(f"Error creating backup of '{self.database_file}': {e}")
                # Continue without backup if it fails, but log the error.

        # Save the new data using atomic write (temporary file + rename)
        temp_file = self.database_file + ".tmp"
        try:
            # File locking (Illustrative, might be overkill for simple CSVs and single processes)
            # This is primarily for concurrent access in multi-process scenarios on Unix-like systems.
            # For Windows, alternative mechanisms like `msvcrt.locking` or higher-level database systems are needed.
            # For cross-platform locking, consider a library like 'portalocker'.
            # import fcntl # Make sure fcntl is imported for posix systems

            if os.name == 'posix': # Only attempt file lock on POSIX systems
                with open(self.database_file, 'a') as f: # Open in append mode to get a file descriptor
                    try:
                        import fcntl # Import fcntl here to avoid ImportError on non-posix systems
                        fcntl.flock(f, fcntl.LOCK_EX) # Acquire an exclusive lock
                        logger.debug(f"Acquired exclusive lock on '{db_path_abs}'.")
                        data.to_csv(temp_file, index=False)
                        os.replace(temp_file, self.database_file) # Atomic replace
                        logger.info(f"Successfully saved {len(data)} records to '{db_path_abs}'.")
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN) # Release the lock
                        logger.debug(f"Released lock on '{db_path_abs}'.")
            else:
                data.to_csv(temp_file, index=False)
                os.replace(temp_file, self.database_file) # Atomic replace
                logger.info(f"Successfully saved {len(data)} records to '{db_path_abs}'.")
            
            # Update last modification time after successful save
            self._last_db_modification_time = datetime.datetime.now()
            # Invalidate caches as database has been updated
            self._cached_database_data = None
            self._cached_monthly_data = None
            self._cached_country_data = None


        except Exception as e:
            logger.error(f"Error saving data to '{db_path_abs}': {e}")
            # Attempt to clean up temp file if something went wrong
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise IOError(f"Failed to save data to database: {e}")


    def save_new_data(self, input_data: pd.DataFrame):
        """
        Saves new data to the database by merging with existing records and updating based on the most recent 'updatetime'.

        This method performs the following operations:
        - Handles empty input data gracefully.
        - Validates the input data for required columns ('symbol', 'updatetime') and critical nulls within these keys.
        - Normalises string columns (trims whitespace, standardises casing) in the input data.
        - Loads the existing database from 'database.csv', also performing basic validation and normalisation.
        - Merges the `input_data` DataFrame with the existing database, aligning columns and logging schema changes.
        - Handles potential duplicate entries by keeping only the most recent record for each 'symbol' based on 'updatetime'.
        - Manages timestamped backups of the existing database, deleting older ones to limit storage.
        - Saves the consolidated and updated DataFrame back to 'database.csv' using atomic file operations.
        - (Illustrative: Mentions file locking to prevent data corruption from concurrent access, if applicable to OS.)

        Args:
            input_data (pd.DataFrame): A pandas DataFrame containing new power plant volume data to be saved.
                                       It must have 'symbol' and 'updatetime' columns for merging and updating.

        Raises:
            IOError: If there are issues reading from or writing to 'database.csv'.
            ValueError: If 'symbol' or 'updatetime' columns are missing from the input_data,
                        or if these critical columns in input_data contain unexpected null values after processing.
        """
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
        if pd.api.types.is_string_dtype(input_data['symbol']): # Check if it's a string column before stripping
            input_data['symbol'] = input_data['symbol'].str.strip().str.lower()
        if not isinstance(input_data['symbol'].dtype, pd.CategoricalDtype):
            logger.debug("Converting 'symbol' column in input data to category type.")
            input_data['symbol'] = input_data['symbol'].astype('category')
        
        if input_data['symbol'].isnull().any():
            logger.error("Critical null values found in 'symbol' column of input data. Cannot proceed with merging.")
            # Null values in 'symbol' are critical because 'symbol' is used as a unique identifier
            # for merging and deduplication. Without a valid symbol, records cannot be correctly
            # tracked or updated in the database, leading to data integrity issues.
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

    def get_data_from_database(self) -> pd.DataFrame:
        """
        Retrieves the full data from the database.csv file.

        This method loads the entire dataset from the configured database file
        ('database.csv') into a pandas DataFrame. It leverages an internal
        helper method (`_load_database`) to handle file loading, potential errors, and basic
        data type conversions (e.g., 'updatetime' to datetime objects).

        Returns:
            pd.DataFrame: A pandas DataFrame containing all records from the database.
                          Returns an empty DataFrame if the database file does not exist
                          or is empty.

        Raises:
            IOError: If there are issues reading the database file (e.g., file not found, permission denied, corruption).
        """
        # Check if cache is valid and up-to-date
        current_db_mtime = None
        if os.path.exists(self.database_file):
            current_db_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(self.database_file))

        # Only use cached database data if it's not None and database hasn't been modified since cache was built
        if self._cached_database_data is not None and self._last_db_modification_time is not None \
           and current_db_mtime is not None and self._last_db_modification_time >= current_db_mtime:
            logger.debug("Returning cached raw database data (no new modifications detected).")
            return self._cached_database_data.copy() # Return a copy to prevent external modification

        logger.debug(f"Attempting to retrieve data from database: '{os.path.abspath(self.database_file)}'.")
        try:
            # Leverage the existing _load_database helper method for consistency and error handling
            data = self._load_database()
            logger.info(f"Successfully retrieved {len(data)} records from database: '{os.path.abspath(self.database_file)}'.")
            
            # Update cache after successful retrieval
            self._cached_database_data = data.copy()
            if current_db_mtime:
                self._last_db_modification_time = current_db_mtime
            else: # If file didn't exist before, but now data is loaded, set current time
                self._last_db_modification_time = datetime.datetime.now()
            
            return data
        except IOError as e:
            logger.error(f"Failed to retrieve data from database '{os.path.abspath(self.database_file)}': {e}")
            self._cached_database_data = None # Invalidate cache on error
            self._last_db_modification_time = None
            raise # Re-raise the exception after logging
        except Exception as e:
            logger.critical(f"An unexpected critical error occurred while retrieving data from '{os.path.abspath(self.database_file)}': {e}")
            self._cached_database_data = None # Invalidate cache on error
            self._last_db_modification_time = None
            raise IOError(f"Critical error during database retrieval: {e}") # Re-raise as IOError for consistency

    def aggregate_data_to_monthly(self, 
                                  groupby_columns: list[str] = None, 
                                  agg_functions: dict = None,
                                  keep_month_as_period: bool = False) -> pd.DataFrame:
        """
        Aggregates the power plant volume data from the database to a monthly level.

        This method loads the complete dataset from 'database.csv', then groups and aggregates it.
        It allows for flexible grouping columns and aggregation functions.

        Args:
            groupby_columns (list[str], optional): A list of column names to group by,
                                                   in addition to the 'month' of 'updatetime'.
                                                   Defaults to ['symbol', 'technology', 'country'].
            agg_functions (dict, optional): A dictionary specifying the aggregation functions.
                                            Keys are new column names and values are pandas aggregation functions
                                            or strings (e.g., 'mean', 'min', 'max').
                                            Defaults to {'average_volume': 'mean', 'min_volume': 'min', 'max_volume': 'max'}.
            keep_month_as_period (bool, optional): If True, the 'month' column will remain as pandas Period objects.
                                                    If False (default), it will be converted to string.

        Returns:
            pd.DataFrame: A DataFrame containing the monthly aggregated data.
                          Returns an empty DataFrame if no data is available or aggregation fails.

        Raises:
            IOError: If there are issues retrieving data from the database.
            ValueError: If required columns ('updatetime', 'volume') or specified `groupby_columns`
                        are missing in the database data after loading.
        """
        # Check if cache is valid and up-to-date and parameters are default
        current_db_mtime = None
        if os.path.exists(self.database_file):
            current_db_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(self.database_file))

        # Only use cache if default parameters are used and database hasn't been modified since cache was built
        if self._cached_monthly_data is not None and self._last_db_modification_time is not None \
           and current_db_mtime is not None and self._last_db_modification_time >= current_db_mtime \
           and groupby_columns is None and agg_functions is None and keep_month_as_period is False:
            logger.debug("Returning cached monthly aggregated data (default parameters, no database modification).")
            return self._cached_monthly_data.copy()

        logger.info("Starting monthly data aggregation.")
        actual_groupby_cols = groupby_columns if groupby_columns is not None else self._DEFAULT_MONTHLY_GROUPBY_COLS
        actual_agg_funcs = agg_functions if agg_functions is not None else self._DEFAULT_MONTHLY_AGG_FUNCTIONS

        # Type validation for inputs
        if not isinstance(actual_groupby_cols, list) or not all(isinstance(col, str) for col in actual_groupby_cols):
            raise TypeError("groupby_columns must be a list of strings.")
        if not isinstance(actual_agg_funcs, dict) or not all(isinstance(k, str) and (isinstance(v, str) or callable(v)) for k, v in actual_agg_funcs.items()):
            raise TypeError("agg_functions must be a dictionary with string keys and string/callable values.")

        try:
            df = self.get_data_from_database() # This method already handles loading and initial datatype conversion
            logger.debug(f"Initial DataFrame shape for monthly aggregation: {df.shape}")
            if df.empty:
                logger.warning("No data in the database to aggregate to monthly level. Returning empty DataFrame.")
                return pd.DataFrame(columns=actual_groupby_cols + ['month'] + list(actual_agg_funcs.keys())) # Return empty with expected columns

            # Explicitly cast 'volume' to float32 early
            if 'volume' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['volume']):
                    logger.warning("Converting 'volume' column to numeric (float32) for aggregation. Non-numeric values will be coerced to NaN.")
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('float32')
                initial_nan_volume_count = df['volume'].isna().sum()
                df = df.dropna(subset=['volume']) # Drop rows where volume could not be converted
                if df['volume'].isna().sum() > initial_nan_volume_count: # Check if new NaNs were introduced
                     logger.warning(f"Dropped {df['volume'].isna().sum() - initial_nan_volume_count} rows due to unparseable 'volume' values after conversion.")

            if df.empty:
                logger.warning("DataFrame became empty after 'volume' conversion and dropping NaN values. Returning empty DataFrame.")
                return pd.DataFrame(columns=actual_groupby_cols + ['month'] + list(actual_agg_funcs.keys()))

            # Cast categorical columns to 'category' if not already
            for col in ['country', 'technology']:
                if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
                    logger.debug(f"Converting column '{col}' to 'category' type.")
                    df[col] = df[col].astype('category')

            # Validate required columns for aggregation (including updatetime, volume, and all groupby columns)
            required_internal_cols = ['updatetime', 'volume'] + actual_groupby_cols
            missing_cols = [col for col in required_internal_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns for monthly aggregation: {missing_cols}. Cannot proceed.")
                raise ValueError(f"Missing required columns for monthly aggregation: {missing_cols}")

            # Create a 'month' column for grouping
            df['month'] = df['updatetime'].dt.to_period('M')
            grouping_keys = actual_groupby_cols + ['month']
            logger.debug(f"Grouping by columns: {grouping_keys}")

            # Perform the aggregation
            # Use observed=False to maintain current pandas behaviour and silence FutureWarning
            # The default value of 'observed' in pandas groupby will change in a future version.
            aggregated_df = df.groupby(grouping_keys, observed=False)['volume'].agg(**actual_agg_funcs).reset_index()
            logger.debug(f"Aggregated DataFrame shape: {aggregated_df.shape}")

            # Convert 'month' Period object back to a string or keep as Period based on option
            if not keep_month_as_period:
                aggregated_df['month'] = aggregated_df['month'].astype(str)
                logger.debug("Converted 'month' column to string type.")
            else:
                logger.debug("Keeping 'month' column as Period type.")
            
            logger.info(f"Successfully aggregated data to monthly level. Resulting records: {len(aggregated_df)}")
            
            # Cache result if default parameters were used
            if groupby_columns is None and agg_functions is None and keep_month_as_period is False:
                self._cached_monthly_data = aggregated_df.copy()
                logger.debug("Monthly aggregation result cached.")

            return aggregated_df

        except IOError as e:
            logger.error(f"Error accessing database for monthly aggregation: {e}")
            raise # Re-raise the exception after logging
        except ValueError as e:
            logger.error(f"Data validation error during monthly aggregation: {e}")
            raise # Re-raise the exception after logging
        except Exception as e:
            logger.critical(f"An unexpected critical error occurred during monthly data aggregation: {e}")
            # Consider returning an empty DataFrame with appropriate columns if it's a non-critical error
            # For now, re-raising as per original logic.
            raise # Re-raise any other unexpected errors

    def aggregate_data_to_country(self, 
                                  groupby_columns: list[str] = None, 
                                  agg_functions: dict = None,
                                  sort_by_columns: list[str] = None) -> pd.DataFrame:
        """
        Aggregates the power plant volume data from the database to a country level.

        This method loads the complete dataset from 'database.csv', then groups and aggregates it.
        It allows for flexible grouping columns and aggregation functions.

        Args:
            groupby_columns (list[str], optional): A list of column names to group by,
                                                   in addition to 'country'.
                                                   Defaults to ['country', 'technology'].
            agg_functions (dict, optional): A dictionary specifying the aggregation functions.
                                            Keys are new column names and values are pandas aggregation functions
                                            or strings (e.g., 'sum', 'mean', 'min', 'max').
                                            Defaults to {'total_volume': 'sum'}.
            sort_by_columns (list[str], optional): A list of column names to sort the final aggregated DataFrame by.
                                                  Defaults to None (no additional sorting).

        Returns:
            pd.DataFrame: A DataFrame containing the country-level aggregated data.
                          Returns an empty DataFrame if no data is available or aggregation fails.

        Raises:
            TypeError: If `groupby_columns`, `agg_functions`, or `sort_by_columns` are not of expected types.
            IOError: If there are issues retrieving data from the database.
            ValueError: If required columns ('volume', 'country', 'technology') or specified `groupby_columns`
                        are missing in the database data after loading, or if `sort_by_columns` are invalid.
        """
        # Check if cache is valid and up-to-date and parameters are default
        current_db_mtime = None
        if os.path.exists(self.database_file):
            current_db_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(self.database_file))

        # Only use cache if default parameters are used and database hasn't been modified since cache was built
        if self._cached_country_data is not None and self._last_db_modification_time is not None \
           and current_db_mtime is not None and self._last_db_modification_time >= current_db_mtime \
           and groupby_columns is None and agg_functions is None and sort_by_columns is None:
            logger.debug("Returning cached country aggregated data (default parameters, no database modification).")
            return self._cached_country_data.copy()

        logger.info("Starting country-level data aggregation.")
        
        actual_groupby_cols = groupby_columns if groupby_columns is not None else self._DEFAULT_COUNTRY_GROUPBY_COLS
        actual_agg_funcs = agg_functions if agg_functions is not None else self._DEFAULT_COUNTRY_AGG_FUNCTIONS

        # Type validation for inputs
        if not isinstance(actual_groupby_cols, list) or not all(isinstance(col, str) for col in actual_groupby_cols):
            raise TypeError("groupby_columns must be a list of strings.")
        if not isinstance(actual_agg_funcs, dict) or not all(isinstance(k, str) and (isinstance(v, str) or callable(v)) for k, v in actual_agg_funcs.items()):
            raise TypeError("agg_functions must be a dictionary with string keys and string/callable values.")
        if sort_by_columns is not None and (not isinstance(sort_by_columns, list) or not all(isinstance(col, str) for col in sort_by_columns)):
            raise TypeError("sort_by_columns must be a list of strings or None.")

        try:
            df = self.get_data_from_database()
            logger.debug(f"Initial DataFrame shape for country aggregation: {df.shape}")
            if df.empty:
                logger.warning("No data in the database to aggregate to country level. Returning empty DataFrame.")
                return pd.DataFrame(columns=actual_groupby_cols + list(actual_agg_funcs.keys())) # Return empty with expected columns

            # Explicitly cast 'volume' to float32 early
            if 'volume' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['volume']):
                    logger.warning("Converting 'volume' column to numeric (float32) for aggregation. Non-numeric values will be coerced to NaN.")
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('float32')
                initial_nan_volume_count = df['volume'].isna().sum()
                df = df.dropna(subset=['volume']) # Drop rows where volume could not be converted
                if df['volume'].isna().sum() > initial_nan_volume_count: # Check if new NaNs were introduced
                     logger.warning(f"Dropped {df['volume'].isna().sum() - initial_nan_volume_count} rows due to unparseable 'volume' values after conversion.")

            if df.empty:
                logger.warning("DataFrame became empty after 'volume' conversion and dropping NaN values. Returning empty DataFrame.")
                return pd.DataFrame(columns=actual_groupby_cols + list(actual_agg_funcs.keys()))

            # Cast categorical columns to 'category' if not already
            for col in ['country', 'technology']:
                if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
                    logger.debug(f"Converting column '{col}' to 'category' type.")
                    df[col] = df[col].astype('category')

            # Validate required columns for aggregation (including updatetime, volume, and all groupby columns)
            required_internal_cols = ['volume', 'country', 'technology'] + actual_groupby_cols
            missing_cols = [col for col in required_internal_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns for country aggregation: {missing_cols}. Cannot proceed.")
                raise ValueError(f"Missing required columns for country aggregation: {missing_cols}")

            # Perform the aggregation
            # Use observed=False to maintain current pandas behaviour and silence FutureWarning
            # The default value of 'observed' in pandas groupby will change in a future version.
            aggregated_df = df.groupby(actual_groupby_cols, observed=False)['volume'].agg(**actual_agg_funcs).reset_index()
            logger.debug(f"Aggregated DataFrame shape: {aggregated_df.shape}")

            # Optional: Sort the final aggregated DataFrame
            if sort_by_columns:
                missing_sort_cols = [col for col in sort_by_columns if col not in aggregated_df.columns]
                if missing_sort_cols:
                    logger.warning(f"Columns specified for sorting not found in aggregated data: {missing_sort_cols}. Skipping sorting.")
                else:
                    aggregated_df = aggregated_df.sort_values(by=sort_by_columns).reset_index(drop=True)
                    logger.debug(f"Aggregated data sorted by: {sort_by_columns}")


            logger.info(f"Successfully aggregated data to country level. Resulting records: {len(aggregated_df)}")
            
            # Cache result if default parameters were used
            if groupby_columns is None and agg_functions is None and sort_by_columns is None:
                self._cached_country_data = aggregated_df.copy()
                logger.debug("Country aggregation result cached.")

            return aggregated_df

        except IOError as e:
            logger.error(f"Error accessing database for country aggregation: {e}")
            raise # Re-raise the exception after logging
        except ValueError as e:
            logger.error(f"Data validation error during country aggregation: {e}")
            raise # Re-raise the exception after logging
        except Exception as e:
            logger.critical(f"An unexpected critical error occurred during country data aggregation: {e}")
            raise # Re-raise any other unexpected errors