import os
import pandas as pd
import numpy as np
from src.utils import setup_logger

# Set up the logger
logger = setup_logger(__name__)

def run_gold_layer_analysis():
    """
    Orchestrates the Gold Layer absence policy breach detection using 
    Business-Day Gaps and Islands sequence analysis with detailed reason tracking.
    """
    logger.info("==========================================================================")
    logger.info("Starting Gold Layer Analysis: Absence Policy Breach Detection")
    logger.info("==========================================================================")
    
    # Load the Silver layer unified dataset
    silver_input_path = os.path.join("data", "silver", "unified_attendance_master.csv")
    if not os.path.exists(silver_input_path):
        logger.error(f"Silver dataset not found at {silver_input_path}. Run Silver pipeline first.")
        raise FileNotFoundError(f"Missing Silver dataset: {silver_input_path}")
        
    logger.info(f"Reading unified Silver dataset from: {silver_input_path}")
    df = pd.read_csv(silver_input_path)
    logger.info(f"Loaded Silver dataset with shape: {df.shape}")
    
    # Ensure gold output directory exists
    gold_dir = os.path.join("data", "gold")
    os.makedirs(gold_dir, exist_ok=True)
    
    # Filter for valid absences (with dayfirst=True for DD/MM/YYYY dates)
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
    
    # Handle boolean/string representation of 'absent' safely
    if df['absent'].dtype == object:
        is_absent = df['absent'].astype(str).str.strip().str.lower().isin(['true', '1', 'y', 'yes'])
    else:
        is_absent = df['absent'] == True
        
    if 'absence_reason' in df.columns:
        reason_col = df['absence_reason'].fillna('').astype(str).str.strip().str.lower().str.replace('_', ' ')
        not_approved_leave = reason_col != 'approved annual leave'
    else:
        not_approved_leave = True
        
    absence_mask = is_absent & not_approved_leave
    abs_df = df[absence_mask].copy()
    
    # Exclude weekends (Saturday=5, Sunday=6) from standard working schedules
    abs_df = abs_df[abs_df['date'].dt.dayofweek < 5]
    logger.info(f"Filtered raw dataset down to {len(abs_df)} valid working-day absence records.")
    
    # Sort chronologically by employee and date
    abs_df = abs_df.sort_values(['employee_num', 'date']).reset_index(drop=True)
    
    # Business-Day Gaps and Islands Sequence Algorithm
    abs_df['bday_int'] = np.busday_count(
        np.datetime64('1970-01-01'), 
        abs_df['date'].values.astype('datetime64[D]')
    )
    
    # Compute cumulative count per employee to form the island group identifier
    abs_df['cum_count'] = abs_df.groupby('employee_num').cumcount()
    abs_df['island_group'] = abs_df['bday_int'] - abs_df['cum_count']
    
    # Aggregate absence spells & capture reasons
    def get_primary_reason(series):
        valid = series.dropna().astype(str).str.strip()
        valid = valid[(valid != '') & (valid.str.lower() != 'nan')]
        if not valid.empty:
            return valid.iloc[0]
        return 'Unspecified'

    spell_summary = abs_df.groupby(['employee_num', 'island_group']).agg(
        spell_start_date=('date', 'min'),
        spell_end_date=('date', 'max'),
        absence_days=('date', 'count'),
        absence_reason=('absence_reason', get_primary_reason)
    ).reset_index()
    
    logger.info(f"Identified {len(spell_summary)} total discrete absence spells across all employees.")
    
    # Format individual spells into readable date ranges including the reason
    spell_summary['spell_detail'] = (
        spell_summary['spell_start_date'].dt.strftime('%Y-%m-%d') + ' to ' + 
        spell_summary['spell_end_date'].dt.strftime('%Y-%m-%d') + 
        ' (' + spell_summary['absence_days'].astype(str) + ' days - Reason: ' + 
        spell_summary['absence_reason'] + ')'
    )
    
    # 5. Enforce Policy Trigger Threshold (3 separate absence periods) & Aggregate Spells
    employee_spell_counts = spell_summary.groupby('employee_num').agg(
        total_absence_periods=('island_group', 'count'),
        absence_periods_detail=('spell_detail', lambda x: ", ".join(x)),
        first_absence_date=('spell_start_date', 'min'),
        last_absence_date=('spell_end_date', 'max')
    ).reset_index()
    
    breach_threshold = 3
    breached_employees = employee_spell_counts[employee_spell_counts['total_absence_periods'] >= breach_threshold].copy()
    
    logger.info(f"Policy evaluation complete: {len(breached_employees)} employees triggered the absence policy ({breach_threshold}+ periods).")
    
    # Enrich with Employee Metadata (Defensive column check)
    metadata_cols = [
        'employee_num', 'first_name', 'last_name', 'department', 
        'division', 'job_title', 'line_manager', 'contract_type', 'location'
    ]
    available_meta_cols = [col for col in metadata_cols if col in df.columns]
    employee_meta = df[available_meta_cols].drop_duplicates(subset=['employee_num'])
    
    # Merge breach summary with employee metadata
    final_breach_report = pd.merge(
        breached_employees,
        employee_meta,
        on='employee_num',
        how='left'
    )
    
    # Reorder columns for professional presentation
    front_cols = [
        'employee_num', 'first_name', 'last_name', 'department', 
        'total_absence_periods', 'absence_periods_detail', 
        'first_absence_date', 'last_absence_date'
    ]
    existing_front_cols = [c for c in front_cols if c in final_breach_report.columns]
    remaining_cols = [c for c in final_breach_report.columns if c not in existing_front_cols]
    final_breach_report = final_breach_report[existing_front_cols + remaining_cols]
    
    # Sort by total absence periods descending
    if 'total_absence_periods' in final_breach_report.columns:
        final_breach_report = final_breach_report.sort_values(by='total_absence_periods', ascending=False)
    
    # Write Gold Output CSV
    output_csv_path = os.path.join(gold_dir, "employee_absence_policy_breaches.csv")
    final_breach_report.to_csv(output_csv_path, index=False)
    logger.info(f"Gold layer breach report successfully written to: {output_csv_path}")
    logger.info(f"Final Gold dataset shape: {final_breach_report.shape}")

if __name__ == "__main__":
    run_gold_layer_analysis()