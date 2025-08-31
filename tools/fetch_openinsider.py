#!/usr/bin/env python3
"""
Fetch OpenInsider Form 4 data and save to engine/data/alternative/

This script downloads insider trading data from OpenInsider and saves it
in a format suitable for the Infinite Money Engine.
"""

import requests
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import io

def fetch_openinsider_data(start_date=None, end_date=None, output_file=None):
    """
    Fetch Form 4 insider trading data from OpenInsider
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format  
        output_file (str): Output file path
    """
    
    # Default to last 30 days if no dates provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # OpenInsider API endpoint for Form 4 filings
    url = "http://openinsider.com/screener"
    
    params = {
        's': '',  # Symbol (empty for all)
        'o': '',  # Owner (empty for all)
        'pl': '',  # Price low
        'ph': '',  # Price high
        'll': '',  # Loss low
        'lh': '',  # Loss high
        'fd': 730,  # Filing date (days back)
        'fdr': start_date,  # Filing date range start
        'fdt': end_date,    # Filing date range end
        'xp': 1,   # Export
        'xs': 1,   # Export
        'vl': '',  # Volume low
        'vh': '',  # Volume high
        'ocl': '', # Ownership change low
        'och': '', # Ownership change high
        'sortcol': 0,  # Sort column
        'cnt': 1000,   # Count
        'page': 1      # Page
    }
    
    try:
        print(f"Fetching insider data from {start_date} to {end_date}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Parse the CSV data
        df = pd.read_csv(io.StringIO(response.text))
        
        # Clean and format the data
        df['Filing Date'] = pd.to_datetime(df['Filing Date'])
        df['Trade Date'] = pd.to_datetime(df['Trade Date'])
        
        # Save to output file
        if not output_file:
            output_file = f"engine/data/alternative/insider_data_{start_date}_{end_date}.csv"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} insider transactions to {output_file}")
        
        return df
        
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def main():
    """Main function to run the script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch OpenInsider Form 4 data')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    # Fetch the data
    df = fetch_openinsider_data(
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=args.output
    )
    
    if df is not None:
        print(f"\nSummary:")
        print(f"Total transactions: {len(df)}")
        print(f"Date range: {df['Filing Date'].min()} to {df['Filing Date'].max()}")
        print(f"Unique companies: {df['Company'].nunique()}")
        print(f"Unique insiders: {df['Owner Name'].nunique()}")

if __name__ == "__main__":
    main()
