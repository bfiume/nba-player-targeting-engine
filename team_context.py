import pandas as pd
import numpy as np

# ── 2025-26 NBA cap figures (hardcoded — public knowledge) ────────────
CAP_2526        = 154_647_000
LUXURY_TAX_2526 = 187_895_000

def build_team_context(master_path='data/master.parquet'):
    df = pd.read_parquet(master_path)

    # Sum current-year salary per team
    payroll = (
        df.groupby('team')['salary']
        .sum()
        .reset_index()
        .rename(columns={'salary': 'total_payroll'})
    )

    payroll['total_payroll_millions']   = (payroll['total_payroll'] / 1_000_000).round(2)
    payroll['cap_millions']             = round(CAP_2526 / 1_000_000, 2)
    payroll['luxury_tax_millions']      = round(LUXURY_TAX_2526 / 1_000_000, 2)
    payroll['cap_space_millions']       = (
        (CAP_2526 - payroll['total_payroll']) / 1_000_000
    ).round(2)
    payroll['over_tax']                 = payroll['total_payroll'] > LUXURY_TAX_2526
    payroll['tax_bill_millions']        = payroll.apply(
        lambda r: round((r['total_payroll'] - LUXURY_TAX_2526) / 1_000_000, 2)
        if r['total_payroll'] > LUXURY_TAX_2526 else 0.0,
        axis=1
    )

    # Player count per team
    player_count = df.groupby('team')['player_name'].count().reset_index()
    player_count.columns = ['team', 'roster_count']
    payroll = payroll.merge(player_count, on='team', how='left')

    payroll.to_csv('data/team_context.csv', index=False)
    print(f"Team context built for {len(payroll)} teams")
    print("\nSample — NYK:")
    print(payroll[payroll['team'] == 'NYK'].to_string(index=False))
    return payroll

if __name__ == '__main__':
    build_team_context()