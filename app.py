import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go

# ── Load data ─────────────────────────────────────────────────────────
df         = pd.read_parquet('data/master.parquet')
hist_df    = pd.read_csv('data/historical.csv')
team_stats = pd.read_csv('data/team_stats.csv')
team_ctx   = pd.read_csv('data/team_context.csv').set_index('team')

POSITIONS      = ['PG', 'SG', 'SF', 'PF', 'C']
TEAMS          = sorted(df['team'].dropna().unique())
CONTRACT_TYPES = ['Expiring', 'Multi-Year', 'Minimum']

BLUE  = '#1D428A'
RED   = '#C8102E'
GREEN = '#007A33'

label_style = {'fontWeight': 'bold', 'fontSize': '13px',
               'marginBottom': '4px', 'display': 'block'}
section_gap = {'marginBottom': '16px'}

PRIORITY_DIMS = {
    'Scoring':    'pct_scoring',
    'Playmaking': 'pct_playmaking',
    'Defense':    'pct_defense',
    'Efficiency': 'pct_efficiency',
    'Rebounding': 'pct_rebounding',
}

RADAR_DIMS = {
    'Scoring':      'pct_scoring',
    'Playmaking':   'pct_playmaking',
    'Defense':      'pct_defense',
    'Efficiency':   'pct_efficiency',
    'Rebounding':   'pct_rebounding',
    'TS%':          'pct_ts',
    'Availability': 'pct_availability',
    'PIE':          'pct_pie',
}

FIT_GAP_DIMS = {
    'Offense':    ('pct_off_rating', 'pct_off_rating'),
    'Defense':    ('pct_def_rating', 'pct_defense'),
    'Scoring':    ('pct_pts',        'pct_scoring'),
    'Playmaking': ('pct_ast',        'pct_playmaking'),
    'Rebounding': ('pct_reb',        'pct_rebounding'),
    'Efficiency': ('pct_efg_pct',    'pct_efficiency'),
}


def score_players(filtered_df, weights):
    filtered_df = filtered_df.copy()
    weight_arr  = np.array([weights.get(k, 0) for k in PRIORITY_DIMS], dtype=float)
    if weight_arr.sum() == 0 or filtered_df.empty:
        filtered_df['score'] = 0.0
        return filtered_df.sort_values('score', ascending=False)
    w_norm      = weight_arr / weight_arr.sum()
    stat_matrix = filtered_df[list(PRIORITY_DIMS.values())].fillna(0).values
    filtered_df['score'] = ((stat_matrix * w_norm).sum(axis=1) * 100).round(1)
    return filtered_df.sort_values('score', ascending=False)


def make_sparkline(y_vals, seasons_list, color, y_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=seasons_list, y=y_vals,
        mode='lines+markers',
        line=dict(color=color, width=2),
        marker=dict(size=6),
        hovertemplate='%{x}: %{y:.1f}<extra></extra>',
    ))
    fig.update_layout(
        height=95,
        margin=dict(l=35, r=10, t=4, b=22),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(size=9), showgrid=False),
        yaxis=dict(
            tickfont=dict(size=9),
            title=dict(text=y_title, font=dict(size=9)),
            showgrid=True, gridcolor='#eee',
        ),
        showlegend=False,
    )
    return fig


# ── Filter panel (shared between Stage 1 and Stage 2) ────────────────
def filter_panel_children():
    return [
        html.H3("Player Search",
                style={'color': BLUE, 'marginTop': '0', 'marginBottom': '14px'}),

        html.Div([
            html.Label("Team Context", style=label_style),
            dcc.Dropdown(id='team-selector',
                         options=[{'label': t, 'value': t} for t in TEAMS],
                         value='NYK', clearable=False),
        ], style=section_gap),

        html.Div(id='cap-context', style={
            'backgroundColor': '#e8eef7',
            'border': f'1px solid {BLUE}',
            'borderRadius': '6px',
            'padding': '10px 12px',
            'fontSize': '12px',
            'marginBottom': '16px',
            'lineHeight': '1.8',
        }),

        html.Div([
            html.Label("Position Need", style=label_style),
            dcc.Checklist(
                id='position-filter',
                options=[{'label': f'  {p}', 'value': p} for p in POSITIONS],
                value=POSITIONS, inline=True,
                inputStyle={'marginRight': '4px'},
                labelStyle={'marginRight': '10px', 'fontSize': '13px'},
            ),
        ], style=section_gap),

        html.Div([
            html.Label("Age Range", style=label_style),
            dcc.RangeSlider(
                id='age-slider', min=19, max=42, step=1, value=[19, 42],
                marks={19: '19', 25: '25', 30: '30', 35: '35', 42: '42'},
                tooltip={'placement': 'bottom', 'always_visible': False},
            ),
        ], style=section_gap),

        html.Div([
            html.Label("Max Salary ($M)", style=label_style),
            dcc.Slider(
                id='salary-slider', min=0, max=60, step=1, value=60,
                marks={0: '$0', 20: '$20M', 40: '$40M', 60: '$60M+'},
                tooltip={'placement': 'bottom', 'always_visible': False},
            ),
        ], style=section_gap),

        html.Div([
            html.Label("Min. Availability %", style=label_style),
            dcc.Slider(
                id='avail-slider', min=0, max=100, step=5, value=0,
                marks={0: '0%', 25: '25%', 50: '50%', 75: '75%', 100: '100%'},
                tooltip={'placement': 'bottom', 'always_visible': False},
            ),
        ], style=section_gap),

        html.Div([
            html.Label("Contract Type", style=label_style),
            dcc.Checklist(
                id='contract-filter',
                options=[{'label': f'  {c}', 'value': c} for c in CONTRACT_TYPES],
                value=CONTRACT_TYPES,
                inputStyle={'marginRight': '4px'},
                labelStyle={'display': 'block', 'fontSize': '13px', 'marginBottom': '4px'},
            ),
        ], style=section_gap),

        html.Hr(style={'borderColor': '#ddd'}),

        html.Div([
            html.H4("Statistical Priorities",
                    style={'color': BLUE, 'margin': '0 0 2px 0'}),
            html.P("0 = ignore  |  10 = most important  |  drives the Score column",
                   style={'fontSize': '11px', 'color': '#888', 'margin': '0 0 10px 0'}),
            *[html.Div([
                html.Label(dim, style={**label_style, 'marginBottom': '2px'}),
                dcc.Slider(
                    id=f'weight-{dim.lower()}',
                    min=0, max=10, step=1, value=5,
                    marks={0: '0', 5: '5', 10: '10'},
                    tooltip={'placement': 'bottom', 'always_visible': False},
                ),
            ], style={'marginBottom': '10px'}) for dim in PRIORITY_DIMS],
        ]),

        html.Hr(style={'borderColor': '#ddd'}),

        html.Button('Reset All Filters', id='reset-button', n_clicks=0,
                    style={
                        'width': '100%', 'padding': '10px',
                        'backgroundColor': BLUE, 'color': 'white',
                        'border': 'none', 'borderRadius': '4px',
                        'cursor': 'pointer', 'fontSize': '13px', 'fontWeight': 'bold',
                    }),
    ]


app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([

    # ── Header ────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H1("NBA Player Targeting Engine",
                    style={'margin': '0', 'color': BLUE, 'fontSize': '26px'}),
            html.P("Define the profile. Discover the fit.",
                   style={'margin': '2px 0 0 0', 'color': '#666', 'fontSize': '13px'}),
        ]),
        # Back to search button — only visible in Stage 2
        html.Button(
            '← Refine Search', id='back-button', n_clicks=0,
            style={
                'display': 'none',   # hidden initially; shown in Stage 2
                'padding': '8px 16px',
                'backgroundColor': 'white',
                'color': BLUE,
                'border': f'2px solid {BLUE}',
                'borderRadius': '4px',
                'cursor': 'pointer',
                'fontSize': '13px',
                'fontWeight': 'bold',
            }
        ),
    ], style={
        'padding': '14px 24px',
        'borderBottom': f'3px solid {BLUE}',
        'backgroundColor': '#f8f9fa',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
    }),

    # ── Stage 1: Guided Setup ─────────────────────────────────────────
    html.Div(
        id='stage-1',
        children=[
            html.Div([

                # Left: filter panel
                html.Div(
                    children=filter_panel_children(),
                    style={
                        'width': '320px', 'minWidth': '320px',
                        'padding': '24px',
                        'backgroundColor': '#f8f9fa',
                        'borderRight': '2px solid #ddd',
                        'overflowY': 'auto',
                        'height': 'calc(100vh - 70px)',
                        'boxSizing': 'border-box',
                    }
                ),

                # Right: welcome / instruction panel
                html.Div([
                    html.Div([
                        html.H2("Welcome to NBA Player Targeting Engine",
                                style={'color': BLUE, 'marginBottom': '12px'}),
                        html.P(
                            "This tool helps NBA front offices identify acquisition targets "
                            "based on a custom player profile. Define your search criteria on "
                            "the left, then click Find Players to see a ranked shortlist with "
                            "detailed visualizations.",
                            style={'fontSize': '15px', 'color': '#444',
                                   'lineHeight': '1.7', 'marginBottom': '32px',
                                   'maxWidth': '560px'}
                        ),

                        html.Div([
                            html.Div([
                                html.Span("①", style={'fontSize': '22px', 'color': BLUE,
                                                       'marginRight': '12px'}),
                                html.Span("Set your Team Context and cap situation",
                                          style={'fontSize': '14px'}),
                            ], style={'display': 'flex', 'alignItems': 'center',
                                      'marginBottom': '16px'}),
                            html.Div([
                                html.Span("②", style={'fontSize': '22px', 'color': BLUE,
                                                       'marginRight': '12px'}),
                                html.Span("Filter by position, age, salary, availability, and contract type",
                                          style={'fontSize': '14px'}),
                            ], style={'display': 'flex', 'alignItems': 'center',
                                      'marginBottom': '16px'}),
                            html.Div([
                                html.Span("③", style={'fontSize': '22px', 'color': BLUE,
                                                       'marginRight': '12px'}),
                                html.Span("Weight your statistical priorities to drive the ranking score",
                                          style={'fontSize': '14px'}),
                            ], style={'display': 'flex', 'alignItems': 'center',
                                      'marginBottom': '40px'}),
                        ]),

                        html.Button(
                            'Find Players →', id='find-button', n_clicks=0,
                            style={
                                'padding': '14px 40px',
                                'backgroundColor': BLUE,
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '6px',
                                'cursor': 'pointer',
                                'fontSize': '16px',
                                'fontWeight': 'bold',
                                'boxShadow': '0 2px 8px rgba(29,66,138,0.3)',
                            }
                        ),
                        html.P("All filters update results instantly after you click Find Players.",
                               style={'fontSize': '11px', 'color': '#aaa', 'marginTop': '12px'}),
                    ], style={
                        'maxWidth': '640px',
                        'margin': 'auto',
                        'paddingTop': '80px',
                    }),
                ], style={
                    'flex': '1',
                    'display': 'flex',
                    'alignItems': 'flex-start',
                    'justifyContent': 'center',
                    'overflowY': 'auto',
                    'height': 'calc(100vh - 70px)',
                }),

            ], style={'display': 'flex', 'height': 'calc(100vh - 70px)'}),
        ],
        style={'display': 'block'}
    ),

    # ── Stage 2: Exploration Mode ─────────────────────────────────────
    html.Div(
        id='stage-2',
        children=[
            html.Div([

                # Collapsible filter sidebar
                html.Div(
                    id='sidebar',
                    children=filter_panel_children(),
                    style={
                        'width': '0px',
                        'minWidth': '0px',
                        'padding': '0px',
                        'overflow': 'hidden',
                        'backgroundColor': '#f8f9fa',
                        'borderRight': '0px solid #ddd',
                        'transition': 'width 0.25s ease, min-width 0.25s ease, padding 0.25s ease',
                        'height': 'calc(100vh - 70px)',
                        'boxSizing': 'border-box',
                    }
                ),

                # Main results area
                html.Div([

                    # Sidebar toggle button
                    html.Div([
                        html.Button(
                            '⚙ Filters', id='sidebar-toggle', n_clicks=0,
                            style={
                                'padding': '6px 14px',
                                'backgroundColor': 'white',
                                'color': BLUE,
                                'border': f'1.5px solid {BLUE}',
                                'borderRadius': '4px',
                                'cursor': 'pointer',
                                'fontSize': '12px',
                                'fontWeight': 'bold',
                                'marginBottom': '12px',
                            }
                        ),
                        html.Span(id='results-header',
                                  style={'fontSize': '16px', 'fontWeight': 'bold',
                                         'color': BLUE, 'marginLeft': '16px'}),
                    ], style={'display': 'flex', 'alignItems': 'center'}),

                    # Score explanation
                    html.Details([
                        html.Summary("How is the Score calculated?",
                                     style={'fontSize': '11px', 'color': BLUE,
                                            'cursor': 'pointer', 'fontWeight': 'bold',
                                            'marginBottom': '4px'}),
                        html.Div([
                            html.P("Each player receives a league percentile rank (0–100) for five "
                                   "stat categories. A score of 75 means that player outperforms 75% "
                                   "of the league in that dimension.", style={'margin': '4px 0'}),
                            html.P("The Overall Score is a weighted average of those five percentiles "
                                   "using your priority slider values as weights. If Defense = 10 and "
                                   "everything else = 0, the Score ranking is identical to the Defense "
                                   "ranking.", style={'margin': '4px 0'}),
                            html.P("Defense percentile = average of steals, blocks, and inverted "
                                   "defensive rating percentiles (lower rating = better defender).",
                                   style={'margin': '4px 0', 'color': '#888'}),
                        ], style={'fontSize': '11px', 'color': '#444',
                                  'backgroundColor': '#f0f4fb', 'padding': '8px 10px',
                                  'borderRadius': '4px'}),
                    ], style={'marginBottom': '8px'}),

                    html.P("Check rows to compare in the charts below. "
                           "Click any row to load the player detail card.",
                           style={'fontSize': '11px', 'color': '#666', 'margin': '0 0 8px 0'}),

                    # V1: Table
                    dash_table.DataTable(
                        id='results-table',
                        columns=[
                            {'name': 'Player',    'id': 'player_name'},
                            {'name': 'Team',      'id': 'team'},
                            {'name': 'Pos',       'id': 'position'},
                            {'name': 'Age',       'id': 'age'},
                            {'name': 'Contract',  'id': 'contract_type'},
                            {'name': 'Sal ($M)',  'id': 'salary_millions'},
                            {'name': 'Yrs',       'id': 'years_remaining'},
                            {'name': 'Guar ($M)', 'id': 'guaranteed_millions'},
                            {'name': 'PPG',       'id': 'ppg'},
                            {'name': 'APG',       'id': 'apg'},
                            {'name': 'RPG',       'id': 'rpg'},
                            {'name': 'TS%',       'id': 'ts_pct_display'},
                            {'name': 'PIE',       'id': 'pie_display'},
                            {'name': 'Avail%',    'id': 'availability_pct'},
                            {'name': 'Score',     'id': 'score'},
                        ],
                        data=[], page_size=10,
                        sort_action='native',
                        row_selectable='multi',
                        selected_rows=[],
                        style_table={'overflowX': 'auto', 'marginBottom': '16px'},
                        style_header={
                            'backgroundColor': BLUE, 'color': 'white',
                            'fontWeight': 'bold', 'fontSize': '11px', 'textAlign': 'center',
                        },
                        style_cell={
                            'fontSize': '11px', 'padding': '6px 8px',
                            'textAlign': 'left', 'whiteSpace': 'normal', 'minWidth': '38px',
                        },
                        style_cell_conditional=[
                            {'if': {'column_id': c}, 'textAlign': 'center'}
                            for c in ['team', 'position', 'age', 'contract_type',
                                      'salary_millions', 'years_remaining',
                                      'guaranteed_millions', 'ppg', 'apg', 'rpg',
                                      'ts_pct_display', 'pie_display',
                                      'availability_pct', 'score']
                        ],
                        style_data_conditional=[
                            {'if': {'row_index': 'odd'}, 'backgroundColor': '#EEF3FB'},
                            {'if': {'column_id': 'score'}, 'fontWeight': 'bold', 'color': BLUE},
                        ],
                    ),

                    # V2 + V3
                    html.Div([
                        html.Div([
                            html.H4("Player Comparison (8-axis)",
                                    style={'color': BLUE, 'margin': '0 0 2px 0'}),
                            html.P("All axes = league percentile. Check rows to overlay up to 3 players.",
                                   style={'fontSize': '11px', 'color': '#888', 'margin': '0 0 2px 0'}),
                            dcc.Graph(id='radar-chart', style={'height': '280px'},
                                      config={'displayModeBar': False}),
                        ], style={'flex': '1', 'minWidth': '0', 'marginRight': '16px'}),

                        html.Div([
                            html.H4("Salary vs. Production",
                                    style={'color': BLUE, 'margin': '0 0 2px 0'}),
                            html.P("Gray = all  |  Blue = matches criteria  |  Red ★ = selected. Y: PIE.",
                                   style={'fontSize': '11px', 'color': '#888', 'margin': '0 0 2px 0'}),
                            dcc.Graph(id='scatter-chart', style={'height': '280px'},
                                      config={'displayModeBar': False}),
                        ], style={'flex': '1', 'minWidth': '0'}),
                    ], style={'display': 'flex', 'marginBottom': '16px'}),

                    # V4 + V5
                    html.Div([
                        html.Div([
                            html.H4("Team Fit Gap",
                                    style={'color': BLUE, 'margin': '0 0 2px 0'}),
                            html.P(id='fit-gap-subtitle',
                                   children="Select a player above to compare against team needs.",
                                   style={'fontSize': '11px', 'color': '#888', 'margin': '0 0 2px 0'}),
                            dcc.Graph(id='fit-gap-chart', style={'height': '280px'},
                                      config={'displayModeBar': False}),
                        ], style={'flex': '1', 'minWidth': '0', 'marginRight': '16px'}),

                        html.Div([
                            html.H4("Player Detail",
                                    style={'color': BLUE, 'margin': '0 0 2px 0'}),
                            html.P("3-year trends for the first selected player.",
                                   style={'fontSize': '11px', 'color': '#888', 'margin': '0 0 2px 0'}),
                            html.Div(id='player-detail-card', style={
                                'height': '280px', 'overflowY': 'auto',
                                'backgroundColor': '#f8f9fa',
                                'border': '1px solid #ddd',
                                'borderRadius': '6px', 'padding': '10px',
                            }),
                        ], style={'flex': '1', 'minWidth': '0'}),
                    ], style={'display': 'flex'}),

                ], style={
                    'flex': '1', 'minWidth': '0', 'padding': '16px 20px',
                    'overflowY': 'auto', 'height': 'calc(100vh - 70px)',
                    'boxSizing': 'border-box',
                }),

            ], style={'display': 'flex', 'height': 'calc(100vh - 70px)'}),
        ],
        style={'display': 'none'}   # hidden until Find Players clicked
    ),

    # Stores
    dcc.Store(id='filtered-store'),
    dcc.Store(id='stage-store', data='setup'),      # 'setup' or 'explore'
    dcc.Store(id='sidebar-store', data='collapsed'), # 'collapsed' or 'expanded'

], style={'fontFamily': 'Arial, sans-serif', 'height': '100vh', 'overflow': 'hidden'})


# ── Callbacks ─────────────────────────────────────────────────────────

# Stage switching
@app.callback(
    Output('stage-store',  'data'),
    Output('stage-1',      'style'),
    Output('stage-2',      'style'),
    Output('back-button',  'style'),
    Input('find-button',   'n_clicks'),
    Input('back-button',   'n_clicks'),
    State('stage-store',   'data'),
    prevent_initial_call=True,
)
def switch_stage(find_clicks, back_clicks, current_stage):
    from dash import ctx
    triggered = ctx.triggered_id

    back_btn_visible = {
        'display': 'block', 'padding': '8px 16px',
        'backgroundColor': 'white', 'color': BLUE,
        'border': f'2px solid {BLUE}', 'borderRadius': '4px',
        'cursor': 'pointer', 'fontSize': '13px', 'fontWeight': 'bold',
    }
    back_btn_hidden = {'display': 'none'}

    if triggered == 'find-button':
        return 'explore', {'display': 'none'}, {'display': 'block'}, back_btn_visible
    else:
        return 'setup', {'display': 'block'}, {'display': 'none'}, back_btn_hidden


# Sidebar toggle
@app.callback(
    Output('sidebar',       'style'),
    Output('sidebar-store', 'data'),
    Input('sidebar-toggle', 'n_clicks'),
    State('sidebar-store',  'data'),
    prevent_initial_call=True,
)
def toggle_sidebar(n_clicks, current_state):
    if current_state == 'collapsed':
        return {
            'width': '270px', 'minWidth': '270px',
            'padding': '18px',
            'overflow': 'auto',
            'backgroundColor': '#f8f9fa',
            'borderRight': '2px solid #ddd',
            'transition': 'width 0.25s ease, min-width 0.25s ease, padding 0.25s ease',
            'height': 'calc(100vh - 70px)',
            'boxSizing': 'border-box',
        }, 'expanded'
    else:
        return {
            'width': '0px', 'minWidth': '0px',
            'padding': '0px',
            'overflow': 'hidden',
            'backgroundColor': '#f8f9fa',
            'borderRight': '0px solid #ddd',
            'transition': 'width 0.25s ease, min-width 0.25s ease, padding 0.25s ease',
            'height': 'calc(100vh - 70px)',
            'boxSizing': 'border-box',
        }, 'collapsed'


# Reset filters
@app.callback(
    Output('position-filter',   'value'),
    Output('age-slider',        'value'),
    Output('salary-slider',     'value'),
    Output('avail-slider',      'value'),
    Output('contract-filter',   'value'),
    Output('weight-scoring',    'value'),
    Output('weight-playmaking', 'value'),
    Output('weight-defense',    'value'),
    Output('weight-efficiency', 'value'),
    Output('weight-rebounding', 'value'),
    Output('results-table',     'selected_rows'),
    Input('reset-button', 'n_clicks'),
    prevent_initial_call=True,
)
def reset_filters(_):
    return POSITIONS, [19, 42], 60, 0, CONTRACT_TYPES, 5, 5, 5, 5, 5, []


# Filter and score
@app.callback(
    Output('filtered-store', 'data'),
    Output('results-header', 'children'),
    Input('position-filter',   'value'),
    Input('age-slider',        'value'),
    Input('salary-slider',     'value'),
    Input('avail-slider',      'value'),
    Input('contract-filter',   'value'),
    Input('weight-scoring',    'value'),
    Input('weight-playmaking', 'value'),
    Input('weight-defense',    'value'),
    Input('weight-efficiency', 'value'),
    Input('weight-rebounding', 'value'),
)
def filter_and_score(positions, age_range, max_salary, min_avail, contracts,
                     w_scr, w_play, w_def, w_eff, w_reb):
    filtered = df.copy()
    if positions:
        filtered = filtered[filtered['position'].isin(positions)]
    else:
        filtered = filtered.iloc[0:0]
    filtered = filtered[(filtered['age'] >= age_range[0]) & (filtered['age'] <= age_range[1])]
    filtered = filtered[filtered['salary_millions'] <= max_salary]
    filtered = filtered[filtered['availability_pct'] >= min_avail]
    if contracts:
        filtered = filtered[filtered['contract_type'].isin(contracts)]
    else:
        filtered = filtered.iloc[0:0]

    weights = {
        'Scoring': w_scr or 0, 'Playmaking': w_play or 0,
        'Defense': w_def or 0, 'Efficiency': w_eff or 0, 'Rebounding': w_reb or 0,
    }
    filtered = score_players(filtered, weights)
    display_cols = [
        'player_name', 'team', 'position', 'age', 'contract_type',
        'salary_millions', 'years_remaining', 'guaranteed_millions',
        'ppg', 'apg', 'rpg', 'ts_pct_display', 'pie_display', 'availability_pct', 'score',
    ]
    result = filtered[display_cols].head(50).round(1)
    return result.to_dict('records'), f"Matching Players  ({len(filtered)} results)"


@app.callback(
    Output('results-table', 'data'),
    Input('filtered-store', 'data'),
)
def update_table(store_data):
    return store_data or []


@app.callback(
    Output('radar-chart', 'figure'),
    Input('results-table', 'selected_rows'),
    Input('filtered-store', 'data'),
)
def update_radar(selected_rows, store_data):
    dims     = list(RADAR_DIMS.keys())
    pct_cols = list(RADAR_DIMS.values())
    colors   = [BLUE, RED, GREEN]
    fig      = go.Figure()

    if not selected_rows or not store_data:
        fig.add_trace(go.Scatterpolar(
            r=[0] * len(dims), theta=dims, fill='toself',
            name='Select a player above', line=dict(color='lightgray'),
        ))
    else:
        records     = pd.DataFrame(store_data)
        player_data = []

        for i, row_idx in enumerate(selected_rows[:3]):
            if row_idx >= len(records):
                continue
            name       = records.iloc[row_idx]['player_name']
            player_row = df[df['player_name'] == name]
            if player_row.empty:
                continue
            player_row = player_row.iloc[0]
            r_vals     = [round(float(player_row.get(col, 0)) * 100, 1) for col in pct_cols]
            color      = colors[i % len(colors)]
            player_data.append((name, r_vals, color))

            fig.add_trace(go.Scatterpolar(
                r=r_vals + [r_vals[0]], theta=dims + [dims[0]],
                fill='toself', name=name,
                line=dict(color=color, width=2), fillcolor=color, opacity=0.2,
                hoverinfo='skip',
            ))

        if player_data:
            for dim_idx, dim in enumerate(dims):
                lines = [f'<b>{dim}</b>']
                for pname, r_vals, _ in player_data:
                    lines.append(f'{pname}: {r_vals[dim_idx]:.0f}')
                tooltip  = '<br>'.join(lines)
                avg_r    = sum(rv[dim_idx] for _, rv, _ in player_data) / len(player_data)
                fig.add_trace(go.Scatterpolar(
                    r=[avg_r], theta=[dim],
                    mode='markers',
                    marker=dict(size=14, opacity=0),
                    hovertemplate=tooltip + '<extra></extra>',
                    showlegend=False,
                ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9), tickvals=[25, 50, 75, 100]),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        showlegend=True,
        legend=dict(font=dict(size=10), orientation='h', yanchor='bottom', y=-0.25),
        margin=dict(l=45, r=45, t=15, b=65),
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
    )
    return fig


@app.callback(
    Output('scatter-chart', 'figure'),
    Input('filtered-store', 'data'),
    Input('results-table',  'selected_rows'),
)
def update_scatter(store_data, selected_rows):
    fig = go.Figure()

    # Determine names to exclude from gray background layer
    matched_names = set()
    sel_names     = set()
    if store_data:
        filtered      = pd.DataFrame(store_data)
        matched_names = set(filtered['player_name'].tolist())
        if selected_rows:
            valid     = [r for r in selected_rows if r < len(filtered)]
            sel_names = set(filtered.iloc[r]['player_name'] for r in valid)

    # Gray: all players NOT in matched set
    bg = df[~df['player_name'].isin(matched_names)]
    fig.add_trace(go.Scatter(
        x=bg['salary_millions'], y=bg['pie_display'],
        mode='markers', marker=dict(color='lightgray', size=5, opacity=0.5),
        text=bg['player_name'],
        hovertemplate='<b>%{text}</b><br>Salary: $%{x}M<br>PIE: %{y:.1f}<extra></extra>',
        name='All players',
    ))
    if store_data:
        # Blue: matched but NOT selected
        blue_names = matched_names - sel_names
        blue_full  = df[df['player_name'].isin(blue_names)]
        fig.add_trace(go.Scatter(
            x=blue_full['salary_millions'], y=blue_full['pie_display'],
            mode='markers', marker=dict(color=BLUE, size=8, opacity=0.85),
            text=blue_full['player_name'],
            hovertemplate='<b>%{text}</b><br>Salary: $%{x}M<br>PIE: %{y:.1f}<extra></extra>',
            name='Matches criteria',
        ))
        if sel_names:
            sel_full = df[df['player_name'].isin(sel_names)]
            fig.add_trace(go.Scatter(
                x=sel_full['salary_millions'], y=sel_full['pie_display'],
                mode='markers+text', marker=dict(color=RED, size=14, symbol='star'),
                text=sel_full['player_name'], textposition='top center',
                textfont=dict(size=10, color=RED),
                hovertemplate='<b>%{text}</b><br>Salary: $%{x}M<br>PIE: %{y:.1f}<extra></extra>',
                name='Selected',
            ))
    fig.update_layout(
        xaxis_title='Salary ($M)', yaxis_title='PIE',
        margin=dict(l=45, r=15, t=15, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=10)),
        hovermode='closest',
    )
    return fig


@app.callback(
    Output('fit-gap-chart',    'figure'),
    Output('fit-gap-subtitle', 'children'),
    Input('team-selector',     'value'),
    Input('results-table',     'selected_rows'),
    Input('filtered-store',    'data'),
)
def update_fit_gap(team, selected_rows, store_data):
    dims     = list(FIT_GAP_DIMS.keys())
    fig      = go.Figure()
    subtitle = f"Showing {team} league percentile by dimension. Select a player to compare."

    team_row = team_stats[team_stats['team'] == team]
    if team_row.empty:
        return fig, subtitle
    team_row  = team_row.iloc[0]
    team_vals = [round(float(team_row.get(FIT_GAP_DIMS[d][0], 0)) * 100, 1) for d in dims]

    bar_colors = [RED if v < 40 else BLUE for v in team_vals]
    fig.add_trace(go.Bar(
        name=f'{team}', x=dims, y=team_vals,
        marker_color=bar_colors, opacity=0.8,
        hovertemplate='<b>%{x}</b><br>%{fullData.name} percentile: %{y:.0f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        name='League avg (50th)', x=dims, y=[50] * len(dims),
        mode='lines', line=dict(color='gray', width=1.5, dash='dash'),
        hoverinfo='skip',
    ))

    player_colors = [GREEN, '#FF8C00', '#9B59B6']
    if selected_rows and store_data:
        records = pd.DataFrame(store_data)
        for i, row_idx in enumerate(selected_rows[:3]):
            if row_idx >= len(records):
                continue
            pname      = records.iloc[row_idx]['player_name']
            player_row = df[df['player_name'] == pname]
            if player_row.empty:
                continue
            player_row  = player_row.iloc[0]
            player_vals = [
                round(float(player_row.get(FIT_GAP_DIMS[d][1], 0)) * 100, 1) for d in dims
            ]
            fig.add_trace(go.Bar(
                name=pname, x=dims, y=player_vals,
                marker_color=player_colors[i % len(player_colors)], opacity=0.65,
                hovertemplate='<b>%{x}</b><br>%{fullData.name} percentile: %{y:.0f}<extra></extra>',
            ))

        player_names = [
            records.iloc[r]['player_name']
            for r in selected_rows[:3] if r < len(records)
        ]
        subtitle = (
            f"{team} (blue/red) vs. {', '.join(player_names)} — "
            f"league percentile by dimension. Red = team weakness below 40th pct."
        )

    fig.update_layout(
        barmode='group',
        yaxis=dict(range=[0, 100], title='League Percentile',
                   tickvals=[0, 25, 50, 75, 100]),
        margin=dict(l=45, r=15, t=15, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=10)),
        hovermode='x unified',
    )
    return fig, subtitle


@app.callback(
    Output('player-detail-card', 'children'),
    Input('results-table',       'selected_rows'),
    Input('filtered-store',      'data'),
)
def update_player_detail(selected_rows, store_data):
    placeholder = html.P(
        "Select a player in the table above to see their 3-year profile.",
        style={'color': '#aaa', 'fontSize': '12px',
               'textAlign': 'center', 'marginTop': '30px'}
    )
    if not selected_rows or not store_data:
        return placeholder

    records = pd.DataFrame(store_data)
    row_idx = selected_rows[0]
    if row_idx >= len(records):
        return placeholder

    pname      = records.iloc[row_idx]['player_name']
    master_row = df[df['player_name'] == pname]
    if master_row.empty:
        return placeholder
    master_row = master_row.iloc[0]

    current_ppg   = float(master_row.get('ppg', 0))
    current_ts    = round(float(master_row.get('ts_pct', 0)) * 100, 1)
    current_avail = float(master_row.get('availability_pct', 0))
    current_sal   = float(master_row.get('salary_millions', 0))
    current_pos   = master_row.get('position', '?')
    current_team  = master_row.get('team', '?')
    contract      = master_row.get('contract_type', '?')
    yrs           = int(master_row.get('years_remaining', 0))
    guar          = float(master_row.get('guaranteed_millions', 0))

    hist_player = hist_df[hist_df['player_name'] == pname].sort_values('season')
    seasons     = list(hist_player['season']) + ['2025-26']
    ppg_vals    = list(hist_player['ppg'].astype(float)) + [current_ppg]
    ts_vals     = [round(float(v) * 100, 1) for v in hist_player['ts_pct']] + [current_ts]
    avail_vals  = list(hist_player['availability_pct'].astype(float)) + [current_avail]

    return html.Div([
        html.Div([
            html.Span(pname, style={'fontWeight': 'bold', 'fontSize': '13px', 'color': BLUE}),
            html.Span(f"  {current_team}  |  {current_pos}",
                      style={'fontSize': '12px', 'color': '#555'}),
        ], style={'marginBottom': '3px'}),
        html.Div(
            f"${current_sal:.1f}M/yr  |  {yrs} yrs left  |  "
            f"${guar:.1f}M guaranteed  |  {contract}",
            style={'fontSize': '11px', 'color': '#666', 'marginBottom': '8px'}
        ),
        html.Div([
            html.P("PPG", style={'fontSize': '10px', 'color': '#888',
                                  'margin': '0', 'fontWeight': 'bold'}),
            dcc.Graph(figure=make_sparkline(ppg_vals, seasons, BLUE, 'PPG'),
                      config={'displayModeBar': False}),
        ]),
        html.Div([
            html.P("TS%", style={'fontSize': '10px', 'color': '#888',
                                  'margin': '4px 0 0 0', 'fontWeight': 'bold'}),
            dcc.Graph(figure=make_sparkline(ts_vals, seasons, GREEN, 'TS%'),
                      config={'displayModeBar': False}),
        ]),
        html.Div([
            html.P("Availability %", style={'fontSize': '10px', 'color': '#888',
                                             'margin': '4px 0 0 0', 'fontWeight': 'bold'}),
            dcc.Graph(figure=make_sparkline(avail_vals, seasons, RED, 'Avail%'),
                      config={'displayModeBar': False}),
        ]),
    ])


@app.callback(
    Output('cap-context', 'children'),
    Input('team-selector', 'value'),
)
def update_cap_context(team):
    if not team or team not in team_ctx.index:
        return "No cap data available."
    row      = team_ctx.loc[team]
    payroll  = row['total_payroll_millions']
    cap      = row['cap_millions']
    tax      = row['luxury_tax_millions']
    space    = row['cap_space_millions']
    over_tax = row['over_tax']
    tax_bill = row['tax_bill_millions']

    space_str = (f"Cap space: ${abs(space):.1f}M over cap"
                 if space < 0 else f"Cap space: ${space:.1f}M available")
    tax_str   = (f"Luxury tax: ${tax_bill:.1f}M over line"
                 if over_tax else f"Tax headroom: ${(tax - payroll):.1f}M")
    tax_color = RED if over_tax else GREEN

    return [
        html.Div([
            html.Span("📋 "),
            html.Span(f"{team} Cap Summary",
                      style={'fontWeight': 'bold', 'color': BLUE, 'fontSize': '12px'}),
        ], style={'marginBottom': '4px'}),
        html.Div(f"Payroll: ${payroll:.1f}M  |  Cap: ${cap:.1f}M", style={'color': '#333'}),
        html.Div(space_str, style={'color': RED if space < 0 else GREEN, 'fontWeight': 'bold'}),
        html.Div(tax_str, style={'color': tax_color, 'fontWeight': 'bold'}),
    ]


server = app.server  # required for Render deployment

if __name__ == '__main__':
    app.run(debug=False)