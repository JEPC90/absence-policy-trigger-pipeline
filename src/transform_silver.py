import os
import pandas as pd
import numpy as np
from src.utils import setup_logger

# Set up the logger
logger = setup_logger(__name__)

def clean_dataframe(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Applies foundational common cleaning routines to input DataFrame:
            a) Standardises column headers (lowercase, stripped, snake_case)
            b) Drops exact duplicate rows
            c) Replaces empty strings and common placeholders with NaN for consistency
            d) Trims whitespace from all string/text columns
            e) Logs a quick null-value audit summary
    """
    logger.info("==========================================================================")
    logger.info(f"Starting Silver Layer Transformation: Stage 1 - Common Cleaning ({table_name})")
    logger.info("==========================================================================") 
    
    initial_row_count = len(df)
    
    # a) Standardise column headers
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    logger.info(f"[{table_name}] Standardised column headers.")
    
    # b) Drop exact duplicate rows
    df = df.drop_duplicates()
    dropped_duplicates = initial_row_count - len(df)
    if dropped_duplicates > 0:
        logger.warning(
            f"[{table_name}] {dropped_duplicates} duplicate rows found and removed."
        )
    else:
        logger.info(f"[{table_name}] No duplicate rows found.")
    
    # c) Replace empty strings with NaN for consistency
    df.replace(r'^\s*$', np.nan, regex=True)
    df = df.replace(['-', 'n/a', 'N/A', 'na', 'NA'], np.nan)
    
    # d) Trim whitespace from all string/text columns
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].str.strip().str.lower()  # Also standardising text to lowercase for consistency
    logger.info(f"[{table_name}] Trimmed whitespace and enforced lowercase consistency across {len(string_cols)} text columns.")
    
    # e) Log null-value audit summary
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        logger.info(f"[{table_name}] Null-value audit complete. Total missing values found: {total_nulls}.")
    else:
        logger.info(f"[{table_name}] Null-value audit complete. Zero missing values found.")
    
    return df

# ==============================================================================
# STAGE 2: SCHEMA EVALUATION & QUARANTINE GATEWAY (DEFINITION)
# ==============================================================================

def evaluate_schema_gateway(df: pd.DataFrame, schema: dict, table_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates rows against strict conditional mandatory constraints and schema rules.
    Rows violating mandatory rules or failing type conversion are routed to quarantine.
    """
    logger.info("===================================================================================")
    logger.info(f"Starting Silver Layer Transformation: Stage 2 - Quarantine Gateway ({table_name})")
    logger.info("===================================================================================")
    
    invalid_mask = pd.Series(False, index=df.index)
    
    if table_name == "Attendance":
        if "date" in df.columns:
            invalid_mask |= df["date"].isnull()
        if "employee_num" in df.columns:
            invalid_mask |= df["employee_num"].isnull()
        
        has_start = "start_time" in df.columns
        has_finish = "finish_time" in df.columns
        has_absent = "absent" in df.columns
        has_reason = "absence_reason" in df.columns
        
        if has_start and has_finish and has_absent:
            zero_time_placeholders = {"00:00:00", "0:00", "00:00", "none", ""}
            
            start_is_blank = (
                df["start_time"].isnull() | 
                df["start_time"].astype(str).str.strip().str.lower().isin(zero_time_placeholders)
            )
            finish_is_blank = (
                df["finish_time"].isnull() | 
                df["finish_time"].astype(str).str.strip().str.lower().isin(zero_time_placeholders)
            )
            blank_times = start_is_blank | finish_is_blank
            
        
            is_absent_y = df["absent"].astype(str).str.strip().str.lower().isin(["y", "true", "1", "yes"])
            
            valid_excused_reasons = {
                    "approved_annual_leave",
                    "minor_illness_(cold/flu)",
                    "headaches/migraines",
                    "back_pain",
                    "acute/recurring_medical_conditions",
                    "gastrointestinal_problems",
                    "other_respiratory_conditions",
                    "accident/injury_work_related",
                    "accident/injury_non_work_related",
                    "care_responsibilities",
                    "musculoskeletal_problems",
                    "stress_anxiety_depression_non_work_related",
                    "stress_anxiety_depression_work_related", 
                    "bereavement", 
                    "maternity_leave", 
                    "paternity_leave", 
                    "training"
                }
            not_approved_leave = ~df["absence_reason"].fillna("").astype(str).str.strip().str.lower().isin(valid_excused_reasons)
        else:
                not_approved_leave = pd.Series(True, index=df.index)
            
        invalid_mask |= (blank_times & is_absent_y & not_approved_leave)

    elif table_name == "Employee":
        if "employee_num" in df.columns:
            invalid_mask |= df["employee_num"].isnull()
            emp_num_parsed = pd.to_numeric(df["employee_num"], errors='coerce')
            invalid_mask |= df["employee_num"].notnull() & (emp_num_parsed.isnull() | (emp_num_parsed < 1000))
        
        if "age" in df.columns:
            age_parsed = pd.to_numeric(df["age"], errors='coerce')
            invalid_mask |= df["age"].notnull() & (age_parsed.isnull() | (age_parsed < 16) | (age_parsed > 100))
    
    for col, expected_type in schema.items():
        if col not in df.columns:
            continue
        
        if expected_type == "int":
            parsed = pd.to_numeric(df[col], errors='coerce')
            col_invalid = parsed.isnull() & df[col].notnull()
        elif expected_type == "numeric":
            parsed = pd.to_numeric(df[col], errors='coerce')
            col_invalid = parsed.isnull() & df[col].notnull()
        elif expected_type == "date":
            parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
            col_invalid = parsed.isnull() & df[col].notnull()
        elif expected_type == "bool":
            valid_states = {0, 1, 0.0, 1.0, True, False, "0", "1", "true", "false", "True", "False", "Y", "N", "y", "n", "YES", "NO", "yes", "no"}
            col_invalid = df[col].notnull() & ~df[col].isin(valid_states)    
        elif expected_type == "str":
            col_invalid = df[col].notnull() & (df[col].astype(str).str.strip() == "")
        else:
            col_invalid = pd.Series(False, index=df.index)
        
        invalid_mask |= col_invalid
    
    valid_subset = df[~invalid_mask].copy()
    quarantine_subset = df[invalid_mask].copy()
    
    logger.info(f"[{table_name}] Quarantine check complete. Valid: {len(valid_subset)}, Quarantined: {len(quarantine_subset)}")
    return valid_subset, quarantine_subset

# ==============================================================================
# STAGE 2b: BESPOKE TEXT NORMALIZATION (DEFINITION)
# ==============================================================================

def normalise_employee_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies bespoke mapping, dictionary replacements, and format normalisations.
    """
    logger.info("[Employee] Starting bespoke text normalisation and standardisation...")
    
    # Gender mapping: 'm' -> 'male', 'f' -> 'female'
    if "gender" in df.columns:
        gender_map = {"m": "male", "f": "female"}
        df["gender"] = df["gender"].replace(gender_map)
        
    # Contract Type mapping: 'ft' -> 'full time'
    if "contract_type" in df.columns:
        contract_map = {"ft": "full time"}
        df["contract_type"] = df["contract_type"].replace(contract_map)
        
    # Job Title normalisation (fixing abbreviations and typos)
    if "job_title" in df.columns:
        job_map = {
            "po": "production operative",
            "production 0perative": "production operative",
            "productionoperative": "production operative"
        }
        df["job_title"] = df["job_title"].replace(job_map)
        
    # 4. Department normalisation (fixing merged word typos)
    if "department" in df.columns:
        dept_map = {
            "machineshop": "machine shop",
            "storesassemby": "stores assembly"
        }
        df["department"] = df["department"].replace(dept_map)
        
    # 5. Location cleaning: remove hyphens ('f-1' -> 'f1', 'f-3' -> 'f3')
    if "location" in df.columns:
        df["location"] = df["location"].str.replace("-", "", regex=False)
        
    logger.info("[Employee] Bespoke text normalisation completed successfully.")
    return df

# ==============================================================================
# STAGE 3: FACT-DIMENSION MERGE & ORPHAN INTERCEPTION (DEFINITION)
# ==============================================================================

def merge_silver_datasets(attendance_df: pd.DataFrame, employee_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a left join between valid attendance fact data and clean employee dimension data
    on 'employee_num', restricts attributes, isolates/quarantines orphans, and logs diagnostics.
    """
    logger.info("===================================================================================")
    logger.info("Starting Silver Layer Transformation: Stage 3 - Fact-Dimension Merge")
    logger.info("===================================================================================")
    
    attendance_df["employee_num"] = pd.to_numeric(attendance_df["employee_num"], errors='coerce').astype('Int64')
    employee_df["employee_num"] = pd.to_numeric(employee_df["employee_num"], errors='coerce').astype('Int64')
    
    desired_employee_columns = [
        "employee_num",
        "first_name",
        "last_name",
        "age",
        "gender",
        "job_title",
        "department",
        "division",
        "location",
        "line_manager",
        "contract_type"
    ]
    
    cols_to_use = [col for col in desired_employee_columns if col in employee_df.columns]
    employee_subset = employee_df[cols_to_use]
    
    merged_df = pd.merge(
        attendance_df,
        employee_subset,
        on="employee_num",
        how="left",
        suffixes=('', '_master'),
        indicator=True
    )
    
    orphan_mask = merged_df["_merge"] == "left_only"
    unmatched_count = orphan_mask.sum()
    
    if unmatched_count > 0:
        logger.warning(f"[Merge] Found {unmatched_count} orphan attendance rows. Running diagnostic breakdown...")
        
        # Diagnostic breakdown: Group orphans by employee_num and log frequencies
        orphans_df = merged_df[orphan_mask].copy()
        orphan_summary = orphans_df.groupby("employee_num", dropna=False).size().reset_index(name="orphan_count")
        logger.warning(f"[Merge] Orphan breakdown by employee_num:\n{orphan_summary.to_string(index=False)}")
        
        # Route orphans to quarantine
        quarantine_path = os.path.join("data", "quarantine", "attendance_orphans_quarantine.csv")
        os.makedirs(os.path.dirname(quarantine_path), exist_ok=True)
        orphans_df.drop(columns=["_merge"]).to_csv(quarantine_path, index=False)
        logger.warning(f"[Merge] Quarantined orphan rows saved to: {quarantine_path}")
        
        # Filter out orphans from the primary Silver dataset
        merged_df = merged_df[~orphan_mask]
    else:
        logger.info("[Merge] 100% match rate: All attendance rows successfully linked.")
        
    merged_df = merged_df.drop(columns=["_merge"])
    
    logger.info(f"Merge complete. Resulting clean unified Silver dataset shape: {merged_df.shape}")
    return merged_df

# ==============================================================================
# MAIN ORCHESTRATION PIPELINE
# ==============================================================================

def main(): 
    try:
        # Define the Bronze file paths
        bronze_attendance_path = os.path.join("data", "bronze", "attendance_raw_data.csv")
        bronze_employee_path = os.path.join("data", "bronze", "employee_raw_data.csv")
        
        # Verify and read attendance data
        logger.info(f"Reading Bronze attendance data from: {bronze_attendance_path}")
        attendance_df = pd.read_csv(bronze_attendance_path)
        
        # Verify and read employee data
        logger.info(f"Reading Bronze employee data from: {bronze_employee_path}")
        employee_df = pd.read_csv(bronze_employee_path)
        
        # ==============================================================================
        # STAGE 1: COMMON CLEANING ROUTINES (INITIATION)
        # ============================================================================== 
        cleaned_attendance = clean_dataframe(attendance_df, "Attendance")
        cleaned_employee = clean_dataframe(employee_df, "Employee")
        
        # ==============================================================================
        # STAGE 2: QUARANTINE FILTERING & NORMALIZATION (INITIATION)
        # ==============================================================================
        os.makedirs(os.path.join("data", "quarantine"), exist_ok=True)
        os.makedirs(os.path.join("data", "silver"), exist_ok=True)
        
        # Define explicit data type mapping schema
        attendance_schema = {
            "day": "str",
            "date": "date",
            "employee_num": "int",
            "fte": "numeric",
            "salary": "numeric",
            "ot1_rate": "numeric",
            "ot2_rate": "numeric",
            "scheduled_rota_hours": "numeric",
            "actual_hours_worked": "str",
            "late_hours": "numeric",
            "absent": "bool",
            "rtw_completed": "bool",
            "flag_holiday_exceeded_160": "bool",
            "ot1": "numeric",
            "ot2": "numeric",
            "scheduled_hours_completed": "numeric",
            "ot_hours_completed": "numeric",
            "total_hours_completed": "numeric",
            "basic_salary_owed": "numeric",
            "ot1_rate_owed": "numeric",
            "ot2_rate_owed": "numeric",
            "hols_owed": "numeric",
            "total_comp_owed": "numeric"
        }
        
        employee_schema = {
            "employee_num": "int",
            "first_name": "str",
            "last_name": "str",
            "age": "int",
            "gender": "str",
            "job_title": "str",
            "department": "str",
            "division": "str",       
            "location": "str",
            "line_manager": "int",
            "contract_type": "str",
            "fte": "numeric",
            "salary": "numeric"
        }
        
        valid_attendance, quarantine_attendance = evaluate_schema_gateway(cleaned_attendance, attendance_schema, "Attendance")
        if len(quarantine_attendance) > 0:
            qu_path = os.path.join("data", "quarantine", "attendance_quarantine.csv")
            quarantine_attendance.to_csv(qu_path, index=False)
            logger.warning(f"[Attendance] Quarantined {len(quarantine_attendance)} rows. Saved to {qu_path}")

        valid_employee, quarantine_employee = evaluate_schema_gateway(cleaned_employee, employee_schema, "Employee")
        if len(quarantine_employee) > 0:
            qu_path = os.path.join("data", "quarantine", "employee_quarantine.csv")
            quarantine_employee.to_csv(qu_path, index=False)
            logger.warning(f"[Employee] Quarantined {len(quarantine_employee)} rows. Saved to {qu_path}")

        if "age" in valid_employee.columns and valid_employee["age"].isnull().sum() > 0:
            logger.warning(f"[Employee] Audit Notice: Found {valid_employee['age'].isnull().sum()} records with missing age values.")
        if "line_manager" in valid_employee.columns and valid_employee["line_manager"].isnull().sum() > 0:
            logger.warning(f"[Employee] Audit Notice: Found {valid_employee['line_manager'].isnull().sum()} records with missing line manager values.")

        valid_employee = normalise_employee_text(valid_employee)
            
        # ==============================================================================
        # STAGE 3: FACT-DIMENSION MERGE (INITIATION)
        # ==============================================================================
        unified_silver_df = merge_silver_datasets(valid_attendance, valid_employee)
        
        silver_output_path = os.path.join("data", "silver", "unified_attendance_master.csv")
        unified_silver_df.to_csv(silver_output_path, index=False)
        logger.info(f"Unified Silver dataset successfully written to: {silver_output_path}")
        
    except Exception as e:
        logger.error(f"Silver Layer Transformation Failed: {e}")
        raise
    
if __name__ == "__main__":  
    main()


    

   