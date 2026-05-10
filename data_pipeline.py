import pandas as pd
import numpy as np
import time
import os
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashplayerbiostats, leaguedashteamstats
from basketball_reference_web_scraper import client

SEASON = '2025-26'

# ── 1. NBA API — per-game stats ───────────────────────────────────────
print("Pulling NBA API per-game stats...")
time.sleep(1)
stats_raw = leaguedashplayerstats.LeagueDashPlayerStats(
    season=SEASON,
    per_mode_detailed='PerGame'
).get_data_frames()[0]
print(f"  {len(stats_raw)} players retrieved")

stats = stats_raw[[
    'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'AGE', 'GP',
    'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
    'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS',
    'OREB', 'DREB', 'FG3A', 'FTA', 'FGA'
]].copy()

stats.columns = [
    'player_id', 'player_name', 'team', 'age', 'games_played',
    'minutes', 'ppg', 'rpg', 'apg', 'spg', 'bpg', 'tov',
    'fg_pct', 'fg3_pct', 'ft_pct', 'plus_minus',
    'oreb', 'dreb', 'fg3a', 'fta', 'fga'
]

# ── 2. NBA API — advanced stats ───────────────────────────────────────
print("Pulling NBA API advanced stats...")
time.sleep(2)
adv_raw = leaguedashplayerstats.LeagueDashPlayerStats(
    season=SEASON,
    per_mode_detailed='PerGame',
    measure_type_detailed_defense='Advanced'
).get_data_frames()[0]
print(f"  {len(adv_raw)} players retrieved")

adv = adv_raw[[
    'PLAYER_ID', 'OFF_RATING', 'DEF_RATING', 'NET_RATING',
    'TS_PCT', 'USG_PCT', 'AST_PCT', 'OREB_PCT', 'DREB_PCT',
    'EFG_PCT', 'PIE'
]].copy()

adv.columns = [
    'player_id', 'off_rating', 'def_rating', 'net_rating',
    'ts_pct', 'usg_pct', 'ast_pct', 'oreb_pct', 'dreb_pct',
    'efg_pct', 'pie'
]

# Merge per-game + advanced
stats = stats.merge(adv, on='player_id', how='left')
print(f"  Per-game + advanced merged: {len(stats)} players")

# ── 3. BBRef — availability + position ───────────────────────────────
print("Pulling BBRef availability and position...")
bbref_raw = pd.DataFrame(client.players_season_totals(season_end_year=2026))

bbref_raw['team_clean'] = (
    bbref_raw['team'].astype(str).str.replace('Team.', '', regex=False)
)

position_map = {
    'POINT_GUARD':    'PG',
    'SHOOTING_GUARD': 'SG',
    'SMALL_FORWARD':  'SF',
    'POWER_FORWARD':  'PF',
    'CENTER':         'C',
}

def parse_position(pos_list):
    if not pos_list or not isinstance(pos_list, list):
        return '?'
    raw = str(pos_list[0]).replace('Position.', '')
    return position_map.get(raw, '?')

bbref_raw['position'] = bbref_raw['positions'].apply(parse_position)

bbref = (
    bbref_raw
    .sort_values('games_played', ascending=False)
    .drop_duplicates(subset='name', keep='first')
)[['name', 'games_played', 'team_clean', 'position']].rename(
    columns={'games_played': 'games_played_bbref'}
).copy()

bbref['availability_pct'] = (bbref['games_played_bbref'] / 82 * 100).round(1)
print(f"  {len(bbref)} players retrieved")

# ── 4. Basketball Reference — salaries ───────────────────────────────
print("Pulling salary data...")
sal_raw = pd.read_html(
    "https://www.basketball-reference.com/contracts/players.html"
)[0]

sal_raw.columns = [
    'rank', 'player_name', 'team', 'salary_2526', 'salary_2627',
    'salary_2728', 'salary_2829', 'salary_2930', 'salary_3031', 'guaranteed'
]

sal = sal_raw[sal_raw['player_name'] != 'Player'].dropna(
    subset=['player_name']
).copy()

def clean_salary(val):
    try:
        return float(str(val).replace('$', '').replace(',', '').strip())
    except:
        return np.nan

sal['salary'] = sal['salary_2526'].apply(clean_salary)

future_cols = ['salary_2627', 'salary_2728', 'salary_2829', 'salary_2930', 'salary_3031']
sal['years_remaining'] = sal[future_cols].apply(
    lambda row: sum(
        1 for v in row
        if pd.notna(v) and str(v).strip() not in ['', 'nan']
    ), axis=1
)

LEAGUE_MIN = 1_500_000

def classify_contract(row):
    if pd.isna(row['salary']) or row['salary'] <= LEAGUE_MIN:
        return 'Minimum'
    elif row['years_remaining'] == 0:
        return 'Expiring'
    else:
        return 'Multi-Year'

sal['contract_type'] = sal.apply(classify_contract, axis=1)

# Clean all future year salary columns
for col in future_cols:
    sal[col] = sal[col].apply(clean_salary)

# Clean guaranteed
sal['guaranteed_clean'] = sal['guaranteed'].apply(clean_salary)
sal['guaranteed_millions'] = (sal['guaranteed_clean'] / 1_000_000).round(2)

# Keep year-by-year columns
sal = sal[[
    'player_name', 'salary', 'years_remaining', 'contract_type',
    'salary_2627', 'salary_2728', 'salary_2829',
    'salary_2930', 'salary_3031', 'guaranteed_millions'
]].copy()
sal = sal.drop_duplicates(subset='player_name', keep='first')
print(f"  {len(sal)} players retrieved")

# ── 5. Merge all sources ──────────────────────────────────────────────
print("Merging all sources...")

df = stats.merge(
    sal[['player_name', 'salary', 'years_remaining', 'contract_type',
         'salary_2627', 'salary_2728', 'salary_2829',
         'salary_2930', 'salary_3031', 'guaranteed_millions']],
    left_on='player_name', right_on='player_name',
    how='left'
)

df = df.merge(
    bbref[['name', 'availability_pct', 'position']],
    left_on='player_name', right_on='name',
    how='left'
).drop(columns=['name'])

# Final dedup — keep highest-minutes row per player
df = (
    df.sort_values('minutes', ascending=False)
    .drop_duplicates(subset='player_name', keep='first')
)

# Fill missing values
df['salary']         = df['salary'].fillna(1_000_000)
df['salary_millions']= (df['salary'] / 1_000_000).round(2)
df['availability_pct']= df['availability_pct'].fillna(
    (df['games_played'] / 82 * 100).round(1)
)
df['contract_type']  = df['contract_type'].fillna('Minimum')
df['years_remaining']= df['years_remaining'].fillna(0).astype(int)
df['guaranteed_millions'] = df['guaranteed_millions'].fillna(0).round(2)
for col in ['salary_2627', 'salary_2728', 'salary_2829', 'salary_2930', 'salary_3031']:
    df[col] = (df[col].fillna(0) / 1_000_000).round(2)
df['position']       = df['position'].fillna('?')

# ── 6. Minutes filter ─────────────────────────────────────────────────
# Keep only players with meaningful playing time
df = df[df['minutes'] >= 10].copy()

# ── 7. Percentile ranks ───────────────────────────────────────────────
print("Computing percentile ranks...")

# Individual percentile columns
pct_map = {
    'ppg':              'pct_scoring',
    'apg':              'pct_playmaking',
    'rpg':              'pct_rebounding',
    'efg_pct':          'pct_efficiency',
    'spg':              'pct_defense_steals',
    'bpg':              'pct_defense_blocks',
    'def_rating':       'pct_def_rating_raw',   # lower is better — inverted below
    'availability_pct': 'pct_availability',
    'ts_pct':           'pct_ts',
    'usg_pct':          'pct_usg',
    'net_rating':       'pct_net_rating',
    'pie':              'pct_pie',
    'off_rating':       'pct_off_rating',
}

for raw_col, pct_col in pct_map.items():
    if raw_col in df.columns:
        df[pct_col] = df[raw_col].rank(pct=True).round(3)

# DEF_RATING: lower is better so invert the percentile
if 'pct_def_rating_raw' in df.columns:
    df['pct_def_rating'] = (1 - df['pct_def_rating_raw']).round(3)
    df.drop(columns=['pct_def_rating_raw'], inplace=True)

# Composite defense: blend steals, blocks, def_rating
df['pct_defense'] = (
    df[['pct_defense_steals', 'pct_defense_blocks', 'pct_def_rating']]
    .mean(axis=1)
    .round(3)
)

# ── Historical data for V5 sparklines ────────────────────────────────
print("Pulling historical stats (2023-24 and 2024-25)...")

historical_seasons = [
    ('2023-24', 2024),
    ('2024-25', 2025),
]

hist_frames = []

for nba_season, bbref_year in historical_seasons:
    time.sleep(2)
    print(f"  Pulling {nba_season}...")

    # Per-game stats
    s = leaguedashplayerstats.LeagueDashPlayerStats(
        season=nba_season,
        per_mode_detailed='PerGame'
    ).get_data_frames()[0]

    s = s[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GP', 'MIN',
            'PTS', 'REB', 'AST']].copy()
    s.columns = ['player_id', 'player_name', 'team', 'games_played',
                 'minutes', 'ppg', 'rpg', 'apg']

    time.sleep(2)

    # Advanced stats for TS%
    adv = leaguedashplayerstats.LeagueDashPlayerStats(
        season=nba_season,
        per_mode_detailed='PerGame',
        measure_type_detailed_defense='Advanced'
    ).get_data_frames()[0]

    adv = adv[['PLAYER_ID', 'TS_PCT']].copy()
    adv.columns = ['player_id', 'ts_pct']
    s = s.merge(adv, on='player_id', how='left')

    time.sleep(2)

    # BBRef availability
    bbref_hist = pd.DataFrame(
        client.players_season_totals(season_end_year=bbref_year)
    )
    bbref_hist = (
        bbref_hist
        .sort_values('games_played', ascending=False)
        .drop_duplicates(subset='name', keep='first')
    )[['name', 'games_played']].copy()
    bbref_hist['availability_pct'] = (
        bbref_hist['games_played'] / 82 * 100
    ).round(1)

    s = s.merge(
        bbref_hist[['name', 'availability_pct']],
        left_on='player_name', right_on='name',
        how='left'
    ).drop(columns=['name'])

    # Dedup and filter
    s = (s.sort_values('minutes', ascending=False)
          .drop_duplicates(subset='player_name', keep='first'))
    s = s[s['minutes'] >= 10].copy()
    s['season'] = nba_season

    hist_frames.append(s)
    print(f"    {len(s)} players")

hist_df = pd.concat(hist_frames, ignore_index=True)
hist_df.to_parquet('data/historical.parquet', index=False)
hist_df.to_csv('data/historical.csv', index=False)

print(f"\nHistorical data saved: {len(hist_df)} rows")
print("Seasons:", hist_df['season'].value_counts().to_dict())

# Quick check — Brunson across both seasons
brunson = hist_df[hist_df['player_name'] == 'Jalen Brunson'][[
    'player_name', 'season', 'ppg', 'apg', 'ts_pct', 'availability_pct'
]]
print("\nBrunson historical:")
print(brunson.to_string(index=False))

# ── Team stats for V4 ─────────────────────────────────────────────────
print("Pulling team stats...")
time.sleep(2)

team_pergame = leaguedashteamstats.LeagueDashTeamStats(
    season=SEASON,
    per_mode_detailed='PerGame'
).get_data_frames()[0]

time.sleep(2)
team_advanced = leaguedashteamstats.LeagueDashTeamStats(
    season=SEASON,
    per_mode_detailed='PerGame',
    measure_type_detailed_defense='Advanced'
).get_data_frames()[0]

# Keep useful columns from each
team_pg = team_pergame[[
    'TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L', 'W_PCT',
    'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
    'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS'
]].copy()
team_pg.columns = [
    'team_id', 'team_name', 'gp', 'w', 'l', 'w_pct',
    'pts', 'reb', 'ast', 'stl', 'blk', 'tov',
    'fg_pct', 'fg3_pct', 'ft_pct', 'plus_minus'
]

team_adv = team_advanced[[
    'TEAM_ID', 'OFF_RATING', 'DEF_RATING', 'NET_RATING',
    'EFG_PCT', 'TS_PCT', 'AST_PCT', 'OREB_PCT', 'DREB_PCT',
    'TM_TOV_PCT', 'PACE', 'PIE'
]].copy()
team_adv.columns = [
    'team_id', 'off_rating', 'def_rating', 'net_rating',
    'efg_pct', 'ts_pct', 'ast_pct', 'oreb_pct', 'dreb_pct',
    'tov_pct', 'pace', 'pie'
]

teams_df = team_pg.merge(team_adv, on='team_id', how='left')

# Add team abbreviation by merging with our existing player data
team_abbrev = (
    df[['team', 'player_id']].copy()
    .merge(
        team_pergame[['TEAM_ID', 'TEAM_NAME']].rename(
            columns={'TEAM_ID': 'team_id', 'TEAM_NAME': 'team_name'}
        ),
        left_on='team', right_on='team_name',
        how='left'
    )[['team', 'team_id']]
    .drop_duplicates()
    .dropna()
)

# Simpler: build abbrev map from nba_api static data
from nba_api.stats.static import teams as nba_teams_static
all_teams = nba_teams_static.get_teams()
abbrev_map = {t['id']: t['abbreviation'] for t in all_teams}
teams_df['team'] = teams_df['team_id'].map(abbrev_map)

# Compute league averages and percentile ranks for each stat
# These are used in V4 to show where a team is weak vs. league average
stat_cols_for_rank = [
    'pts', 'reb', 'ast', 'stl', 'blk',
    'fg_pct', 'fg3_pct', 'off_rating', 'def_rating',
    'net_rating', 'efg_pct', 'ts_pct', 'pace', 'pie'
]

for col in stat_cols_for_rank:
    if col in teams_df.columns:
        if col == 'def_rating':
            # Lower is better for defense
            teams_df[f'pct_{col}'] = (
                1 - teams_df[col].rank(pct=True)
            ).round(3)
        else:
            teams_df[f'pct_{col}'] = (
                teams_df[col].rank(pct=True)
            ).round(3)

# League averages for V4 comparison
league_avgs = {col: teams_df[col].mean() for col in stat_cols_for_rank
               if col in teams_df.columns}
league_avgs_df = pd.DataFrame([league_avgs])
league_avgs_df.to_csv('data/league_averages.csv', index=False)

teams_df.to_parquet('data/team_stats.parquet', index=False)
teams_df.to_csv('data/team_stats.csv', index=False)

print(f"  Team stats saved: {len(teams_df)} teams")
print("\nNYK team stats:")
print(teams_df[teams_df['team'] == 'NYK'][[
    'team', 'w', 'l', 'pts', 'off_rating', 'def_rating',
    'net_rating', 'efg_pct', 'ts_pct', 'pie',
    'pct_off_rating', 'pct_def_rating', 'pct_net_rating'
]].to_string(index=False))

# ── 8. Save ───────────────────────────────────────────────────────────
# Display versions (multiplied by 100 for readability)
df['ts_pct_display'] = (df['ts_pct'] * 100).round(1)
df['pie_display']    = (df['pie'] * 100).round(1)
os.makedirs('data', exist_ok=True)
df.to_parquet('data/master.parquet', index=False)
df.to_csv('data/master.csv', index=False)

print(f"\nDone. Master dataset: {len(df)} players, {len(df.columns)} columns")
print("\nNew advanced stat columns available:")
print([c for c in df.columns if c not in [
    'player_id','player_name','team','age','games_played','minutes',
    'ppg','rpg','apg','spg','bpg','tov','fg_pct','fg3_pct','ft_pct',
    'plus_minus','oreb','dreb','fg3a','fta','fga','salary',
    'salary_millions','years_remaining','contract_type','availability_pct','position'
]])

print("\nSample — Knicks players:")
knicks = df[df['team'] == 'NYK'][[
    'player_name', 'position', 'ppg',
    'salary_millions', 'salary_2627', 'salary_2728',
    'guaranteed_millions', 'contract_type', 'years_remaining'
]].sort_values('ppg', ascending=False)
print(knicks.to_string(index=False))