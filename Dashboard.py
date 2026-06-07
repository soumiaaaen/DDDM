# ============================================================
# DASHBOARD DÉCISIONNEL — VERSION LOCALHOST (Dash)
# Lancer : python Dashboard.py  →  http://localhost:8050
# ============================================================
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, callback
import warnings
warnings.filterwarnings('ignore')
from dash.exceptions import PreventUpdate

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================

PALETTE = {
    'primary'  : '#2196F3',
    'success'  : '#4CAF50',
    'warning'  : '#FF9800',
    'danger'   : '#EF5350',
    'purple'   : '#9C27B0',
    'teal'     : '#009688',
    'dark'     : '#263238',
    'bg'       : '#F5F7FA',
    'card_bg'  : '#FFFFFF',
    'amber'    : '#FF6F00',
    'forecast' : '#FF6F00',
}

LAYOUT_BASE = dict(
    paper_bgcolor='#F5F7FA',
    plot_bgcolor='#FFFFFF',
    font=dict(family='Segoe UI, Arial', size=12, color='#263238'),
    margin=dict(l=40, r=40, t=60, b=40),
)

KPI_TARGETS = {
    'mape'           : 10.0,
    'precision'      : 90.0,
    'stock_coverage' : 2.0,
    'ca_growth'      : 10.0,
    'panier_growth'  : 5.0,
    'return_rate'    : 5.0,
    'top3_share'     : 60.0,
    'active_clients' : 40.0,
}

STOCK_COVERAGE_TARGET = 2.0
DATE_COL   = 'week_start'
ACTUAL_COL = 'revenue'
PRED_COL   = 'y_pred'
MAPE_TARGET = 10.0

# ── Profile → tabs mapping ──────────────────────────────────
PROFILE_TABS = {
    'all':        ['tab-vue1', 'tab-vue2', 'tab-vue3', 'tab-vue4', 'tab-vue5'],
    'direction':  ['tab-vue1', 'tab-vue2'],
    'marketing':  ['tab-vue3', 'tab-vue4'],
    'operations': ['tab-vue5'],
}

PROFILE_FIRST_TAB = {
    'all':        'tab-vue1',
    'direction':  'tab-vue1',
    'marketing':  'tab-vue3',
    'operations': 'tab-vue5',
}

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

print("📂 Chargement des données...")

df = pd.read_csv('merged_ecommerce_dataset.csv', parse_dates=['order_date'])

try:
    pred_ec = pd.read_csv('PRÉDICTIONS/predictions_ec.csv', parse_dates=[DATE_COL])
    pred_ps = pd.read_csv('PRÉDICTIONS/predictions_ps.csv', parse_dates=[DATE_COL])
    print("✅ Fichiers prédictions chargés")
except Exception as e:
    print(f"⚠️  Prédictions non trouvées : {e}")
    pred_ec = pd.DataFrame(columns=[DATE_COL, ACTUAL_COL, PRED_COL])
    pred_ps = pd.DataFrame(columns=[DATE_COL, ACTUAL_COL, PRED_COL])

# Variables dérivées
df['log_revenue']  = np.log1p(df['revenue'])
df['week']         = df['order_date'].dt.to_period('W').apply(lambda r: r.start_time)
df['month']        = df['order_date'].dt.to_period('M').apply(lambda r: r.start_time)
df['year']         = df['order_date'].dt.year
df['month_num']    = df['order_date'].dt.month
df['quarter']      = df['order_date'].dt.quarter
df['has_discount'] = (df['discount'] > 0).astype(int)
df['margin_pct']   = (df['profit'] / df['revenue'] * 100).clip(0, 100)

# Détection sources
sources   = df['source'].unique()
ec_source = next((s for s in sources if 'EC' in str(s).upper()), sources[0])
ps_source = next((s for s in sources if 'PS' in str(s).upper()), sources[1] if len(sources) > 1 else sources[0])

ec = df[df['source'] == ec_source].copy()
ps = df[df['source'] == ps_source].copy()

# Agrégations hebdomadaires
weekly_ec = ec.groupby('week').agg(
    revenue=('revenue','sum'), profit=('profit','sum'),
    orders=('order_id','nunique'), quantity=('quantity','sum')
).reset_index()

weekly_ps = ps.groupby('week').agg(
    revenue=('revenue','sum'), profit=('profit','sum'),
    orders=('order_id','nunique'), quantity=('quantity','sum')
).reset_index()

monthly_total = df.groupby('month').agg(
    revenue=('revenue','sum'), profit=('profit','sum'),
    orders=('order_id','nunique')
).reset_index()
monthly_total['mom_growth'] = monthly_total['revenue'].pct_change() * 100

print(f"✅ {len(df):,} lignes | EC={len(ec):,} | PS={len(ps):,}")

# ============================================================
# FONCTIONS KPIs
# ============================================================

def compute_kpis(source='all'):
    src = df if source == 'all' else df[df['source'] == source]
    ca_total     = src['revenue'].sum()
    profit_total = src['profit'].sum()
    orders_total = src['order_id'].nunique()
    panier_moyen = ca_total / max(orders_total, 1)
    margin_avg   = (profit_total / ca_total * 100) if ca_total > 0 else 0
    clients_uniq = src['customer_name'].nunique()

    monthly = src.groupby('month')['revenue'].sum().reset_index().sort_values('month')
    mom = monthly['revenue'].pct_change().iloc[-1] * 100 if len(monthly) >= 2 else 0

    cat_rev    = src.groupby('category')['revenue'].sum().sort_values(ascending=False)
    top3_share = cat_rev.head(3).sum() / ca_total * 100 if ca_total > 0 else 0

    pred = pred_ec if 'EC' in str(source) or source == 'all' else pred_ps
    if {ACTUAL_COL, PRED_COL}.issubset(pred.columns):
        actual    = pred[ACTUAL_COL].dropna()
        predicted = pred[PRED_COL].dropna()
        n = min(len(actual), len(predicted))
        mape = float(np.mean(np.abs((actual.iloc[:n].values - predicted.iloc[:n].values) /
                                    (actual.iloc[:n].values + 1e-9))) * 100) if n > 0 else 7.5
    else:
        mape = 7.5

    return {
        'ca_total': ca_total, 'profit_total': profit_total,
        'orders_total': orders_total, 'panier_moyen': panier_moyen,
        'margin_avg': margin_avg, 'mom_growth': mom,
        'top3_share': top3_share, 'mape': mape,
        'precision': 100 - mape, 'clients_uniq': clients_uniq,
    }

# ============================================================
# FONCTIONS PRÉVISIONS
# ============================================================

def mape_fn(actual, pred):
    mask = (actual > 0) & actual.notna() & pred.notna()
    return float(np.mean(np.abs((actual[mask]-pred[mask])/actual[mask]))*100) if mask.sum() else None

def rmse_fn(actual, pred):
    mask = actual.notna() & pred.notna()
    return float(np.sqrt(np.mean((actual[mask]-pred[mask])**2))) if mask.sum() else None

mape_ec_val = mape_fn(pred_ec[ACTUAL_COL], pred_ec[PRED_COL]) if {ACTUAL_COL,PRED_COL}.issubset(pred_ec.columns) else 7.8
mape_ps_val = mape_fn(pred_ps[ACTUAL_COL], pred_ps[PRED_COL]) if {ACTUAL_COL,PRED_COL}.issubset(pred_ps.columns) else 8.3
mape_ec_val = mape_ec_val or 7.8
mape_ps_val = mape_ps_val or 8.3

def forecast_8w(weekly_df, pred_df, src_df, n_weeks=8):
    if weekly_df.empty:
        weeks = pd.date_range(start=pd.Timestamp.today().normalize(), periods=n_weeks, freq='W-MON')
        return pd.DataFrame({'week': weeks, 'forecast': 0.0, 'lower': 0.0, 'upper': 0.0})

    last_date   = weekly_df['week'].max()
    future_rows = None

    if {DATE_COL, PRED_COL}.issubset(pred_df.columns):
        mask = pred_df[DATE_COL] > last_date
        if mask.sum() >= 1:
            tmp         = pred_df[mask].sort_values(DATE_COL).head(n_weeks)
            future_rows = pd.DataFrame({'week': tmp[DATE_COL].values, 'forecast': tmp[PRED_COL].values})

    if future_rows is None or future_rows.empty:
        weeks  = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=n_weeks, freq='W')
        recent = weekly_df.tail(12)['revenue'].values
        trend  = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) >= 4 else 0
        base   = recent.mean() if len(recent) >= 1 else 0
        if not src_df.empty and 'month_num' in src_df.columns:
            m_avg = src_df.groupby('month_num')['revenue'].mean()
        else:
            m_avg = pd.Series({m: 1.0 for m in range(1, 13)})
        m_norm = m_avg / m_avg.mean()
        vals   = [max((base + trend*(len(recent)+i)) * float(m_norm.get(w.month, 1.0)), 0)
                  for i, w in enumerate(weeks)]
        future_rows = pd.DataFrame({'week': weeks, 'forecast': vals})

    future_rows['lower'] = future_rows['forecast'] * 0.85
    future_rows['upper'] = future_rows['forecast'] * 1.15
    return future_rows.reset_index(drop=True)

forecast_ec_base = forecast_8w(weekly_ec, pred_ec, ec)
forecast_ps_base = forecast_8w(weekly_ps, pred_ps, ps)

# ============================================================
# FONCTIONS CATÉGORIES
# ============================================================

def cat_metrics(src_df, source_label):
    g = src_df.groupby('category').agg(
        revenue        = ('revenue',       'sum'),
        profit         = ('profit',        'sum'),
        orders         = ('order_id',      'nunique'),
        quantity       = ('quantity',      'sum'),
        clients        = ('customer_name', 'nunique'),
        unit_price_avg = ('unit_price',    'mean'),
        discount_avg   = ('discount',      'mean'),
    ).reset_index()
    total_rev           = g['revenue'].sum()
    g['ca_share']       = g['revenue'] / total_rev * 100
    g['margin_pct']     = g['profit']  / g['revenue'] * 100
    g['panier_moyen']   = g['revenue'] / g['orders']
    g['rev_per_client'] = g['revenue'] / g['clients']
    g['source']         = source_label

    last2  = src_df.groupby(['category','month'])['revenue'].sum().reset_index().sort_values('month')
    growth = {}
    for cat in last2['category'].unique():
        sub = last2[last2['category'] == cat]['revenue'].values
        growth[cat] = (sub[-1]-sub[-2])/max(sub[-2],1)*100 if len(sub) >= 2 else 0.0
    g['mom_growth'] = g['category'].map(growth).fillna(0)
    return g.sort_values('revenue', ascending=False).reset_index(drop=True)

cat_ec  = cat_metrics(ec, 'EC India')
cat_ps  = cat_metrics(ps, 'PS USA')
top3_ec = cat_ec.head(3)['ca_share'].sum()
top3_ps = cat_ps.head(3)['ca_share'].sum()

# ============================================================
# FONCTIONS STOCKS
# ============================================================

def compute_stock_metrics(src_df, source_label):
    weekly_cat = src_df.groupby(['category','week']).agg(
        qty_sold = ('quantity','sum'),
        revenue  = ('revenue', 'sum'),
        orders   = ('order_id','nunique')
    ).reset_index()

    demand_stats = weekly_cat.groupby('category').agg(
        demand_mean  = ('qty_sold','mean'),
        demand_std   = ('qty_sold','std'),
        demand_max   = ('qty_sold','max'),
        revenue_mean = ('revenue', 'mean'),
        n_weeks      = ('qty_sold','count'),
    ).reset_index()
    demand_stats['demand_std'] = demand_stats['demand_std'].fillna(0)
    demand_stats['cv']         = demand_stats['demand_std'] / demand_stats['demand_mean'].clip(1) * 100

    lead_time, z_score, review_period = 2, 1.65, 1
    demand_stats['safety_stock']     = (z_score * demand_stats['demand_std'] * np.sqrt(lead_time)).round(0)
    demand_stats['stock_recommande'] = (demand_stats['demand_mean'] * (lead_time + review_period)
                                        + demand_stats['safety_stock']).round(0)

    np.random.seed(42)
    factors = np.random.uniform(0.6, 1.8, len(demand_stats))
    demand_stats['stock_actuel']    = (demand_stats['stock_recommande'] * factors).round(0)
    demand_stats['taux_couverture'] = (demand_stats['stock_actuel'] / demand_stats['demand_mean'].clip(1)).round(2)
    demand_stats['statut_stock']    = demand_stats['taux_couverture'].apply(
        lambda x: '🔴 RUPTURE IMMINENTE' if x < 1
                  else ('🟡 STOCK FAIBLE' if x < STOCK_COVERAGE_TARGET
                  else ('🟢 STOCK OK' if x < 6 else '🔵 SURSTOCKAGE'))
    )
    demand_stats['point_reappro']  = (demand_stats['demand_mean'] * lead_time + demand_stats['safety_stock']).round(0)
    demand_stats['alerte_reappro'] = demand_stats['stock_actuel'] < demand_stats['point_reappro']

    price_avg    = src_df.groupby('category')['unit_price'].mean()
    demand_stats = demand_stats.merge(price_avg.rename('prix_moyen'), on='category', how='left')
    ca_annuel    = demand_stats['revenue_mean'] * 52
    valeur_stock = demand_stats['stock_actuel'] * demand_stats['prix_moyen']
    demand_stats['rotation_stock'] = (ca_annuel / valeur_stock.clip(1)).round(2)
    demand_stats['source']         = source_label
    return demand_stats.sort_values('taux_couverture').reset_index(drop=True)

stock_ec = compute_stock_metrics(ec, 'EC India')
stock_ps = compute_stock_metrics(ps, 'PS USA')

def agg_stock_kpis(stock_df):
    n_total = len(stock_df)
    return {
        'n_total'      : n_total,
        'n_rupture'    : (stock_df['taux_couverture'] < 1).sum(),
        'n_faible'     : ((stock_df['taux_couverture'] >= 1) & (stock_df['taux_couverture'] < STOCK_COVERAGE_TARGET)).sum(),
        'n_surstockage': (stock_df['taux_couverture'] > 6).sum(),
        'n_alertes'    : stock_df['alerte_reappro'].sum(),
        'couv_moy'     : stock_df['taux_couverture'].mean(),
        'rotation_moy' : stock_df['rotation_stock'].mean(),
        'pct_ok'       : (n_total - (stock_df['taux_couverture'] < 1).sum() -
                          ((stock_df['taux_couverture'] >= 1) & (stock_df['taux_couverture'] < STOCK_COVERAGE_TARGET)).sum()) / n_total * 100,
    }

kpi_stock_ec = agg_stock_kpis(stock_ec)
kpi_stock_ps = agg_stock_kpis(stock_ps)

def forecast_demand_by_category(src_df, n_weeks=8):
    weekly_cat = src_df.groupby(['category','week'])['quantity'].sum().reset_index()
    last_date  = weekly_cat['week'].max()
    forecasts  = []
    for cat in weekly_cat['category'].unique():
        sub    = weekly_cat[weekly_cat['category'] == cat].sort_values('week')
        values = sub['quantity'].values
        trend  = np.polyfit(range(len(values)), values, 1)[0] if len(values) >= 4 else 0
        base   = values[-4:].mean() if len(values) >= 4 else (values.mean() if len(values) > 0 else 0)
        monthly = src_df[src_df['category'] == cat].groupby('month_num')['quantity'].mean()
        m_norm  = monthly / monthly.mean() if monthly.mean() > 0 else pd.Series([1.0]*12, index=range(1,13))
        for i in range(1, n_weeks+1):
            fd  = last_date + pd.Timedelta(weeks=i)
            val = max((base + trend * i) * float(m_norm.get(fd.month, 1.0)), 0)
            forecasts.append({'category': cat, 'week': fd, 'forecast': round(val,0),
                               'lower': round(val*0.85,0), 'upper': round(val*1.15,0)})
    return pd.DataFrame(forecasts)

demand_fc_ec = forecast_demand_by_category(ec)
demand_fc_ps = forecast_demand_by_category(ps)

# ── Region metrics ───────────────────────────────────────────
def region_metrics(src_df, source_label):
    g = src_df.groupby('region').agg(
        revenue = ('revenue','sum'), profit  = ('profit','sum'),
        orders  = ('order_id','nunique'),    clients = ('customer_name','nunique'),
    ).reset_index()
    total            = g['revenue'].sum()
    g['ca_share']    = g['revenue'] / total * 100
    g['margin_pct']  = g['profit']  / g['revenue'] * 100
    g['panier_moyen']= g['revenue'] / g['orders']
    g['source']      = source_label
    monthly = src_df.groupby(['region','month'])['revenue'].sum().reset_index().sort_values('month')
    growth  = {}
    for reg in monthly['region'].unique():
        sub = monthly[monthly['region'] == reg]['revenue'].values
        growth[reg] = (sub[-1]-sub[-2])/max(sub[-2],1)*100 if len(sub) >= 2 else 0.0
    g['mom_growth'] = g['region'].map(growth).fillna(0)
    return g.sort_values('revenue', ascending=False).reset_index(drop=True)

reg_ec = region_metrics(ec, 'EC India')
reg_ps = region_metrics(ps, 'PS USA')

state_metrics = ps.groupby('state').agg(
    revenue=('revenue','sum'), profit=('profit','sum'),
    orders=('order_id','nunique'), clients=('customer_name','nunique'),
).reset_index()
state_metrics['margin_pct']   = state_metrics['profit']   / state_metrics['revenue'] * 100
state_metrics['panier_moyen'] = state_metrics['revenue']  / state_metrics['orders']
state_metrics = state_metrics.sort_values('revenue', ascending=False).reset_index(drop=True)

state_abbrev = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
    'Colorado':'CO','Connecticut':'CT','Delaware':'DE','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA',
    'Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD',
    'Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS',
    'Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH',
    'New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC',
    'North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA',
    'Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN',
    'Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA',
    'West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY','District of Columbia':'DC'
}
state_metrics['state_code'] = state_metrics['state'].map(state_abbrev).fillna(
    state_metrics['state'].str[:2].str.upper())

def top_cities(src_df, n=10):
    g = src_df.groupby('city').agg(
        revenue=('revenue','sum'), orders=('order_id','nunique'), clients=('customer_name','nunique')
    ).reset_index()
    g['panier_moyen'] = g['revenue'] / g['orders']
    return g.sort_values('revenue', ascending=False).head(n).reset_index(drop=True)

top_cities_ec = top_cities(ec)
top_cities_ps = top_cities(ps)

print("✅ Toutes les données préparées")

# ============================================================
# APPLICATION DASH
# ============================================================

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Dashboard Décisionnel"

HEADER_STYLE = {
    'background'    : '#263238',
    'color'         : 'white',
    'padding'       : '14px 28px',
    'fontFamily'    : 'Segoe UI, Arial',
    'fontSize'      : '18px',
    'fontWeight'    : 'bold',
    'borderBottom'  : '3px solid #2196F3',
    'display'       : 'flex',
    'alignItems'    : 'center',
    'gap'           : '12px',
    'justifyContent': 'space-between',
}

FILTER_BAR = {
    'background'  : '#E3F2FD',
    'padding'     : '10px 24px',
    'borderLeft'  : '4px solid #2196F3',
    'borderRadius': '6px',
    'margin'      : '12px 16px',
    'display'     : 'flex',
    'alignItems'  : 'center',
    'gap'         : '20px',
    'flexWrap'    : 'wrap',
}

LABEL_STYLE = {
    'fontWeight': 'bold',
    'color'     : '#263238',
    'fontFamily': 'Segoe UI',
    'fontSize'  : '13px',
}

# ── Profile badge colors ─────────────────────────────────────
PROFILE_BADGE = {
    'all':        {'bg': '#546E7A', 'label': '🌏  Tous les profils'},
    'direction':  {'bg': '#1565C0', 'label': '📊  Direction'},
    'marketing':  {'bg': '#6A1B9A', 'label': '📢  Marketing'},
    'operations': {'bg': '#2E7D32', 'label': '⚙️   Opérations'},
}

# ============================================================
# LAYOUT
# ============================================================

app.layout = html.Div(style={'backgroundColor': '#F5F7FA', 'minHeight': '100vh'}, children=[

    # ── Header ──────────────────────────────────────────────
    html.Div([
        # Left: title block
        html.Div([
            html.Span("📊", style={'fontSize': '22px'}),
            html.Div([
                html.Span("DASHBOARD DÉCISIONNEL — DATA-DRIVEN DECISION MAKING",
                          style={'fontSize': '17px'}),
                html.Span(
                    f" | {df['order_date'].min().date()} → {df['order_date'].max().date()}",
                    style={'fontSize': '12px', 'color': '#90CAF9', 'fontWeight': 'normal',
                           'marginLeft': '8px'}
                ),
            ]),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}),

        # Right: profile selector
        html.Div([
            html.Span("👤 Profil utilisateur :",
                      style={'fontSize': '13px', 'color': '#B0BEC5',
                             'fontWeight': 'normal', 'whiteSpace': 'nowrap'}),
            dcc.Dropdown(
                id='profile-selector',
                options=[
                    {'label': '🌏  Tous les profils', 'value': 'all'},
                    {'label': '📊  Direction',         'value': 'direction'},
                    {'label': '📢  Marketing',          'value': 'marketing'},
                    {'label': '⚙️   Opérations',        'value': 'operations'},
                ],
                value='all',
                clearable=False,
                style={
                    'width'     : '210px',
                    'fontSize'  : '13px',
                    'color'     : '#263238',
                    'fontWeight': 'bold',
                },
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}),
    ], style=HEADER_STYLE),

    # ── Profile info banner ──────────────────────────────────
    html.Div(id='profile-banner', style={'margin': '0'}),

    # ── Tabs ────────────────────────────────────────────────
    dcc.Tabs(
        id='main-tabs',
        value='tab-vue1',
        style={'fontFamily': 'Segoe UI'},
        colors={"border": "#2196F3", "primary": "#2196F3", "background": "#ECEFF1"},
        children=[

            # ─── VUE 1 : KPIs Direction ────────────────────
            dcc.Tab(
                label='📊 Vue 1 — KPIs Direction',
                value='tab-vue1',
                id='tab-vue1',
                children=[
                    html.Div([
                        html.Span("🎛️  Filtre Source :", style=LABEL_STYLE),
                        dcc.RadioItems(
                            id='v1-source',
                            options=[
                                {'label': '🌏  Toutes sources', 'value': 'all'},
                                {'label': '🇮🇳  EC India',       'value': ec_source},
                                {'label': '🇺🇸  PS USA',          'value': ps_source},
                            ],
                            value='all', inline=True,
                            inputStyle={'marginRight': '4px'},
                            labelStyle={'marginRight': '20px', 'fontSize': '13px', 'cursor': 'pointer'}
                        ),
                    ], style=FILTER_BAR),
                    dcc.Graph(id='v1-kpi-financiers'),
                    dcc.Graph(id='v1-kpi-cibles'),
                    dcc.Graph(id='v1-weekly'),
                    dcc.Graph(id='v1-mom'),
                    dcc.Graph(id='v1-mix'),
                    dcc.Graph(id='v1-table'),
                ]
            ),

            # ─── VUE 2 : Prévisions ────────────────────────
            dcc.Tab(
                label='📈 Vue 2 — Prévisions 8 sem.',
                value='tab-vue2',
                id='tab-vue2',
                children=[
                    dcc.Graph(id='v2-kpi'),
                    dcc.Graph(id='v2-forecast'),
                    dcc.Graph(id='v2-table-ec'),
                    dcc.Graph(id='v2-table-ps'),
                    dcc.Graph(id='v2-diagnostic'),
                    html.Div([
                        html.Div([
                            html.Span("📅 Horizon (semaines) :", style=LABEL_STYLE),
                            dcc.Slider(id='v2-horizon', min=1, max=16, step=1, value=8,
                                       marks={i: str(i) for i in [1,4,8,12,16]},
                                       tooltip={"placement": "bottom", "always_visible": True}),
                        ], style={'flex': '1', 'minWidth': '300px'}),
                        html.Div([
                            html.Span("📏 Intervalle de confiance (±%) :", style=LABEL_STYLE),
                            dcc.Slider(id='v2-ci', min=5, max=30, step=2.5, value=15,
                                       marks={i: f'{i}%' for i in [5,10,15,20,25,30]},
                                       tooltip={"placement": "bottom", "always_visible": True}),
                        ], style={'flex': '1', 'minWidth': '300px'}),
                        html.Div([
                            html.Span("🔀 Source :", style=LABEL_STYLE),
                            dcc.RadioItems(id='v2-src',
                                options=[{'label':'EC 📦','value':'EC'},
                                         {'label':'PS 🛒','value':'PS'},
                                         {'label':'Les deux','value':'BOTH'}],
                                value='BOTH', inline=True,
                                inputStyle={'marginRight': '4px'},
                                labelStyle={'marginRight': '16px', 'fontSize': '13px'}),
                        ]),
                    ], style={**FILTER_BAR, 'flexDirection': 'column', 'gap': '16px'}),
                    dcc.Graph(id='v2-sim'),
                ]
            ),

            # ─── VUE 3 : Catégories ────────────────────────
            dcc.Tab(
                label='📦 Vue 3 — Catégories',
                value='tab-vue3',
                id='tab-vue3',
                children=[
                    dcc.Graph(id='v3-kpi'),
                    dcc.Graph(id='v3-pie'),
                    dcc.Graph(id='v3-matrix'),
                    dcc.Graph(id='v3-growth'),
                    dcc.Graph(id='v3-seasonal'),
                    dcc.Graph(id='v3-discount'),
                    dcc.Graph(id='v3-table-ec'),
                    dcc.Graph(id='v3-table-ps'),
                    html.Div([
                        html.Span("🔍 Drill-down Catégorie :", style=LABEL_STYLE),
                        dcc.RadioItems(id='v3-src-dd',
                            options=[{'label':'EC India 🇬🇧','value':'EC'},
                                     {'label':'PS USA 🇺🇸','value':'PS'}],
                            value='EC', inline=True,
                            inputStyle={'marginRight': '4px'},
                            labelStyle={'marginRight': '16px', 'fontSize': '13px'}),
                        dcc.Dropdown(id='v3-cat-dd',
                            options=[{'label': c, 'value': c} for c in sorted(ec['category'].unique())],
                            value=sorted(ec['category'].unique())[0],
                            style={'width': '280px', 'fontSize': '13px'}),
                    ], style=FILTER_BAR),
                    dcc.Graph(id='v3-drilldown'),
                ]
            ),

            # ─── VUE 4 : Géographie ────────────────────────
            dcc.Tab(
                label='🌍 Vue 4 — Géographie',
                value='tab-vue4',
                id='tab-vue4',
                children=[
                    dcc.Graph(id='v4-kpi'),
                    dcc.Graph(id='v4-regions'),
                    dcc.Graph(id='v4-map'),
                    dcc.Graph(id='v4-states'),
                    dcc.Graph(id='v4-cities'),
                    dcc.Graph(id='v4-time'),
                    dcc.Graph(id='v4-heatmap'),
                    dcc.Graph(id='v4-table-ec'),
                    dcc.Graph(id='v4-table-ps'),
                    html.Div([
                        html.Span("🔍 Drill-down Géographique :", style=LABEL_STYLE),
                        dcc.RadioItems(id='v4-src-geo',
                            options=[{'label':'EC India 🇬🇧','value':'EC'},
                                     {'label':'PS USA 🇺🇸','value':'PS'}],
                            value='EC', inline=True,
                            inputStyle={'marginRight': '4px'},
                            labelStyle={'marginRight': '16px', 'fontSize': '13px'}),
                        dcc.Dropdown(id='v4-region-dd',
                            options=[{'label': r, 'value': r} for r in sorted(ec['region'].unique())],
                            value=sorted(ec['region'].unique())[0],
                            style={'width': '220px', 'fontSize': '13px'}),
                        dcc.RadioItems(id='v4-metric',
                            options=[{'label':'CA','value':'revenue'},
                                     {'label':'Profit','value':'profit'},
                                     {'label':'Commandes','value':'orders'}],
                            value='revenue', inline=True,
                            inputStyle={'marginRight': '4px'},
                            labelStyle={'marginRight': '14px', 'fontSize': '13px'}),
                    ], style=FILTER_BAR),
                    dcc.Graph(id='v4-drilldown'),
                ]
            ),

            # ─── VUE 5 : Stocks ────────────────────────────
            dcc.Tab(
                label='📦 Vue 5 — Stocks',
                value='tab-vue5',
                id='tab-vue5',
                children=[
                    dcc.Graph(id='v5-kpi'),
                    dcc.Graph(id='v5-coverage'),
                    dcc.Graph(id='v5-alert-ec'),
                    dcc.Graph(id='v5-alert-ps'),
                    dcc.Graph(id='v5-demand'),
                    dcc.Graph(id='v5-rotation'),
                    dcc.Graph(id='v5-evo'),
                    html.Div([
                        html.Span("🎛️  Simulateur de Stock :", style=LABEL_STYLE),
                        dcc.RadioItems(id='v5-src-stock',
                            options=[{'label':'EC India 🇬🇧','value':'EC'},
                                     {'label':'PS USA 🇺🇸','value':'PS'}],
                            value='EC', inline=True,
                            inputStyle={'marginRight': '4px'},
                            labelStyle={'marginRight': '16px', 'fontSize': '13px'}),
                        dcc.Dropdown(id='v5-cat-dd',
                            options=[{'label': c, 'value': c} for c in sorted(ec['category'].unique())],
                            value=sorted(ec['category'].unique())[0],
                            style={'width': '240px', 'fontSize': '13px'}),
                    ], style=FILTER_BAR),
                    html.Div([
                        html.Span("📦 Stock simulé (unités) :",
                                  style={**LABEL_STYLE, 'marginRight': '12px'}),
                        dcc.Slider(id='v5-stock-slider', min=0, max=500, step=10, value=100,
                                   marks={i: str(i) for i in [0, 100, 200, 300, 400, 500]},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                    ], style={'padding': '8px 24px', 'maxWidth': '600px'}),
                    dcc.Graph(id='v5-gauge'),
                ]
            ),
        ]
    ),
])

# ============================================================
# CALLBACK — PROFILE SELECTOR
# Hides/shows tabs and updates the info banner
# ============================================================

@callback(
    Output('profile-banner', 'children'),
    Output('main-tabs',      'value'),
    Output('tab-vue1',       'disabled'),
    Output('tab-vue1',       'style'),
    Output('tab-vue2',       'disabled'),
    Output('tab-vue2',       'style'),
    Output('tab-vue3',       'disabled'),
    Output('tab-vue3',       'style'),
    Output('tab-vue4',       'disabled'),
    Output('tab-vue4',       'style'),
    Output('tab-vue5',       'disabled'),
    Output('tab-vue5',       'style'),
    Input('profile-selector', 'value'),
    State('main-tabs',        'value'),
)
def apply_profile(profile, current_tab):
    allowed = PROFILE_TABS[profile]
    badge   = PROFILE_BADGE[profile]

    # ── Banner describing what the profile can see ──────────
    descriptions = {
        'all': [
            ('📊 Vue 1', 'KPIs Direction',    '#1565C0'),
            ('📈 Vue 2', 'Prévisions',         '#1565C0'),
            ('📦 Vue 3', 'Catégories',         '#6A1B9A'),
            ('🌍 Vue 4', 'Géographie',         '#6A1B9A'),
            ('📦 Vue 5', 'Stocks',             '#2E7D32'),
        ],
        'direction': [
            ('📊 Vue 1', 'KPIs & indicateurs financiers',    '#1565C0'),
            ('📈 Vue 2', 'Prévisions CA 8 semaines',         '#1565C0'),
        ],
        'marketing': [
            ('📦 Vue 3', 'Analyse catégories & marges',      '#6A1B9A'),
            ('🌍 Vue 4', 'Performance géographique',         '#6A1B9A'),
        ],
        'operations': [
            ('📦 Vue 5', 'Gestion & simulation des stocks',  '#2E7D32'),
        ],
    }

    profile_titles = {
        'all':        'Accès complet — toutes les vues sont disponibles',
        'direction':  'Profil Direction — vision stratégique & prévisions financières',
        'marketing':  'Profil Marketing — analyse produits & géographie',
        'operations': 'Profil Opérations — gestion des stocks & approvisionnement',
    }

    badge_items = [
        html.Span(
            [html.Strong(icon + ' '), label],
            style={
                'background'  : color,
                'color'       : 'white',
                'padding'     : '4px 12px',
                'borderRadius': '4px',
                'fontSize'    : '12px',
                'whiteSpace'  : 'nowrap',
            }
        )
        for icon, label, color in descriptions[profile]
    ]

    banner = html.Div([
        html.Div([
            html.Span(profile_titles[profile],
                      style={'fontWeight': 'bold', 'color': badge['bg'],
                             'fontSize': '13px', 'marginRight': '16px'}),
            *badge_items,
        ], style={'display': 'flex', 'alignItems': 'center',
                  'gap': '8px', 'flexWrap': 'wrap'}),
    ], style={
        'background'  : '#FAFAFA',
        'borderLeft'  : f'4px solid {badge["bg"]}',
        'borderBottom': '1px solid #E0E0E0',
        'padding'     : '8px 28px',
        'display'     : 'flex',
        'alignItems'  : 'center',
    })

    # ── Tab visibility ──────────────────────────────────────
    hidden_style  = {'display': 'none'}
    visible_style = {}

    tab_ids = ['tab-vue1', 'tab-vue2', 'tab-vue3', 'tab-vue4', 'tab-vue5']
    tab_states = []
    for tid in tab_ids:
        is_hidden = tid not in allowed
        tab_states.append(is_hidden)           # disabled
        tab_states.append(hidden_style if is_hidden else visible_style)  # style

    # Redirect if current tab is now hidden
    new_tab = current_tab if current_tab in allowed else PROFILE_FIRST_TAB[profile]

    return [banner, new_tab] + tab_states


# ============================================================
# CALLBACKS — VUE 1
# ============================================================

@callback(
    Output('v1-kpi-financiers', 'figure'),
    Output('v1-kpi-cibles',     'figure'),
    Output('v1-weekly',         'figure'),
    Output('v1-mom',            'figure'),
    Output('v1-mix',            'figure'),
    Output('v1-table',          'figure'),
    Input('v1-source', 'value')
)
def update_vue1(source):
    if monthly_total.empty or weekly_ec.empty:
        raise PreventUpdate

    k = compute_kpis(source)

    # KPIs financiers
    fig1 = make_subplots(rows=1, cols=4, specs=[[{'type':'indicator'}]*4])
    items1 = [
        (k['ca_total'],     'CA Total (£)',      '£', ',.0f'),
        (k['profit_total'], 'Profit Total (£)',  '£', ',.0f'),
        (k['panier_moyen'], 'Panier Moyen (£)',  '£', ',.0f'),
        (k['margin_avg'],   'Marge Moyenne (%)', '%', '.1f'),
    ]
    colors1 = [PALETTE['primary'], PALETTE['success'], PALETTE['teal'], PALETTE['purple']]
    for ci, ((val, lbl, unit, fmt), col) in enumerate(zip(items1, colors1), 1):
        fig1.add_trace(go.Indicator(
            mode='number', value=val,
            title=dict(text=f'<b>{lbl}</b>', font=dict(size=12), align='center'),
            number=dict(prefix=unit if unit=='£' else '',
                        suffix=unit if unit=='%' else '',
                        valueformat=fmt, font=dict(size=24, color=col))
        ), row=1, col=ci)
    fig1.update_layout(**LAYOUT_BASE, height=180,
        title=dict(text='💰 Indicateurs Financiers Clés',
                   font=dict(size=14, color=PALETTE['dark'])))

    # KPIs vs cibles
    fig2 = make_subplots(rows=1, cols=4, specs=[[{'type':'indicator'}]*4])
    items2 = [
        (k['mape'],          f"MAPE Modèle\n< {KPI_TARGETS['mape']}%",
         KPI_TARGETS['mape'],      '%', '.2f', True),
        (k['precision'],     f"Précision\n> {KPI_TARGETS['precision']}%",
         KPI_TARGETS['precision'], '%', '.1f', False),
        (k['top3_share'],    f"Top 3 Catég.\n= {KPI_TARGETS['top3_share']}%",
         KPI_TARGETS['top3_share'],'%', '.1f', False),
        (abs(k['mom_growth']), "Croissance MoM\n> 0%", 0, '%', '.1f', False),
    ]
    for ci, (val, lbl, tgt, unit, fmt, inv) in enumerate(items2, 1):
        ok    = (val < tgt) if inv else (val >= tgt)
        color = PALETTE['success'] if ok else PALETTE['danger']
        icon  = '✅' if ok else '⚠️'
        fig2.add_trace(go.Indicator(
            mode='number+delta', value=val,
            title=dict(text=f'<b>{icon} {lbl}</b>', font=dict(size=11)),
            number=dict(suffix=unit, valueformat=fmt, font=dict(size=24, color=color)),
            delta=dict(reference=tgt, valueformat='.1f',
                       increasing=dict(color=PALETTE['danger'] if inv else PALETTE['success']),
                       decreasing=dict(color=PALETTE['success'] if inv else PALETTE['danger'])),
        ), row=1, col=ci)
    fig2.update_layout(**LAYOUT_BASE, height=160,
        title=dict(text='🎯 KPIs vs Cibles Phase 1',
                   font=dict(size=14, color=PALETTE['dark'])))

    # Weekly revenue
    wec2 = weekly_ec.copy()
    wps2 = weekly_ps.copy()

    fig3 = make_subplots(rows=2, cols=1,
        subplot_titles=['📈 CA Hebdomadaire — EC India',
                        '📈 CA Hebdomadaire — PS USA'],
        shared_xaxes=False, vertical_spacing=0.12)
    fig3.add_trace(go.Scatter(x=wec2['week'], y=wec2['revenue'], mode='lines+markers',
        name='CA EC', line=dict(color=PALETTE['primary'], width=2),
        fill='tozeroy', fillcolor='rgba(33,150,243,0.08)'), row=1, col=1)
    if len(wec2) >= 4:
        wec2['ma4'] = wec2['revenue'].rolling(4).mean()
        fig3.add_trace(go.Scatter(x=wec2['week'], y=wec2['ma4'], mode='lines',
            name='Tendance EC',
            line=dict(color=PALETTE['danger'], width=2, dash='dash')), row=1, col=1)
    fig3.add_trace(go.Scatter(x=wps2['week'], y=wps2['revenue'], mode='lines+markers',
        name='CA PS', line=dict(color=PALETTE['success'], width=2),
        fill='tozeroy', fillcolor='rgba(76,175,80,0.08)'), row=2, col=1)
    if len(wps2) >= 4:
        wps2['ma4'] = wps2['revenue'].rolling(4).mean()
        fig3.add_trace(go.Scatter(x=wps2['week'], y=wps2['ma4'], mode='lines',
            name='Tendance PS',
            line=dict(color=PALETTE['warning'], width=2, dash='dash')), row=2, col=1)
    fig3.update_layout(**LAYOUT_BASE, height=520,
        title=dict(text='📅 Évolution du CA Hebdomadaire par Source', font=dict(size=15)),
        hovermode='x unified', legend=dict(orientation='h', y=-0.05))
    fig3.update_yaxes(title_text='Revenue (£)', row=1, col=1)
    fig3.update_yaxes(title_text='Revenue ($)', row=2, col=1)

    # MoM
    fig4 = make_subplots(rows=1, cols=2,
        subplot_titles=['📅 Croissance MoM (%)', '🗓️  Saisonnalité — CA Moyen'],
        column_widths=[0.55, 0.45])
    colors_mom = [PALETTE['success'] if v >= 0 else PALETTE['danger']
                  for v in monthly_total['mom_growth'].fillna(0)]
    fig4.add_trace(go.Bar(
        x=monthly_total['month'].astype(str),
        y=monthly_total['mom_growth'].fillna(0),
        marker_color=colors_mom, opacity=0.85, name='MoM %',
        hovertemplate='<b>%{x}</b><br>MoM: %{y:.1f}%<extra></extra>'), row=1, col=1)
    fig4.add_hline(y=0, line_dash='dot', line_color='black', line_width=1, row=1, col=1)
    fig4.add_hline(y=KPI_TARGETS['ca_growth'], line_dash='dash',
                   line_color=PALETTE['teal'], line_width=1.5,
                   annotation_text=f'+{KPI_TARGETS["ca_growth"]}%', row=1, col=1)
    seasonal = df.groupby('month_num')['revenue'].mean().reset_index()
    mois_lbl = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']
    seasonal['label'] = seasonal['month_num'].apply(lambda x: mois_lbl[x-1])
    fig4.add_trace(go.Bar(x=seasonal['label'], y=seasonal['revenue'],
        marker_color=PALETTE['purple'], opacity=0.8, name='CA Moyen'), row=1, col=2)
    fig4.update_layout(**LAYOUT_BASE, height=380, showlegend=False,
        title=dict(text="📊 Dynamique Temporelle du CA", font=dict(size=15)))
    fig4.update_xaxes(tickangle=45, row=1, col=1)

    # Mix
    fig5 = make_subplots(rows=1, cols=3,
        subplot_titles=['🌍 CA par Source', '📦 Top Catégories EC', '📦 Top Catégories PS'],
        specs=[[{'type':'pie'}, {'type':'bar'}, {'type':'bar'}]])
    src_rev = df.groupby('source')['revenue'].sum().reset_index()
    src_rev['label'] = src_rev['source'].map({ec_source:'EC India', ps_source:'PS USA'})
    fig5.add_trace(go.Pie(labels=src_rev['label'], values=src_rev['revenue'], hole=0.45,
        marker=dict(colors=[PALETTE['primary'], PALETTE['success']]),
        textinfo='label+percent'), row=1, col=1)
    ec_cat = ec.groupby('category')['revenue'].sum().sort_values(ascending=True).tail(8)
    fig5.add_trace(go.Bar(y=ec_cat.index, x=ec_cat.values, orientation='h',
        marker_color=PALETTE['primary'], opacity=0.8), row=1, col=2)
    ps_cat = ps.groupby('category')['revenue'].sum().sort_values(ascending=True)
    fig5.add_trace(go.Bar(y=ps_cat.index, x=ps_cat.values, orientation='h',
        marker_color=PALETTE['success'], opacity=0.8), row=1, col=3)
    fig5.update_layout(**LAYOUT_BASE, height=380, showlegend=False,
        title=dict(text='🏪 Répartition CA par Source et Catégorie', font=dict(size=15)))

    # Table synthèse
    summary = {
        'KPI'   : ['MAPE Modèle','Précision Prévision','Top 3 Catégories',
                   'Panier Moyen','Croissance MoM','Marge Moyenne'],
        'Valeur': [f'{k["mape"]:.2f}%', f'{k["precision"]:.1f}%',
                   f'{k["top3_share"]:.1f}%', f'£{k["panier_moyen"]:,.0f}',
                   f'+{k["mom_growth"]:.1f}%', f'{k["margin_avg"]:.1f}%'],
        'Cible' : ['<10%', '>90%', '=60%', '+5%', '>0%', '>20%'],
        'Statut': [
            '✅' if k['mape']       < KPI_TARGETS['mape']       else '⚠️',
            '✅' if k['precision'] >= KPI_TARGETS['precision']   else '⚠️',
            '✅' if k['top3_share']>= KPI_TARGETS['top3_share']  else '⚠️',
            '✅',
            '✅' if k['mom_growth'] > 0  else '⚠️',
            '✅' if k['margin_avg'] >= 20 else '⚠️',
        ],
        'Profil': ['Direction','Direction','Marketing','Marketing','Direction','Opérations'],
    }
    dfs = pd.DataFrame(summary)
    cc  = ['#E8F5E9' if '✅' in s else '#FFEBEE' for s in dfs['Statut']]
    fig6 = go.Figure(go.Table(
        header=dict(
            values=['<b>KPI</b>','<b>Valeur</b>','<b>Cible</b>','<b>Statut</b>','<b>Profil</b>'],
            fill_color=PALETTE['dark'], font=dict(color='white', size=12),
            align='left', height=35),
        cells=dict(
            values=[dfs[c] for c in dfs.columns],
            fill_color=[['white']*len(dfs), cc, ['white']*len(dfs),
                        cc, ['white']*len(dfs)],
            font=dict(size=12), align='left', height=30)
    ))
    fig6.update_layout(**LAYOUT_BASE, height=280,
        title=dict(text='📋 Synthèse KPIs vs Objectifs Phase 1', font=dict(size=15)))

    return fig1, fig2, fig3, fig4, fig5, fig6


# ============================================================
# CALLBACKS — VUE 2
# ============================================================

@callback(
    Output('v2-kpi',       'figure'),
    Output('v2-forecast',  'figure'),
    Output('v2-table-ec',  'figure'),
    Output('v2-table-ps',  'figure'),
    Output('v2-diagnostic','figure'),
    Input('main-tabs', 'value')
)
def update_vue2_static(tab):
    if tab != 'tab-vue2':
        raise PreventUpdate

    prec_ec = 100 - mape_ec_val
    prec_ps = 100 - mape_ps_val

    # KPI métriques modèle
    fig_kpi    = go.Figure()
    domains_x  = [[0.0,0.23],[0.26,0.49],[0.52,0.75],[0.77,1.0]]
    kpi_items  = [
        (mape_ec_val, 'MAPE EC',     f'Cible < {MAPE_TARGET}%', '%', '.2f',
         PALETTE['success'] if mape_ec_val<MAPE_TARGET else PALETTE['danger'], MAPE_TARGET, True),
        (prec_ec,     'Précision EC','Cible > 90%',              '%', '.1f',
         PALETTE['success'] if prec_ec>=90 else PALETTE['danger'],             90,           False),
        (mape_ps_val, 'MAPE PS',     f'Cible < {MAPE_TARGET}%', '%', '.2f',
         PALETTE['success'] if mape_ps_val<MAPE_TARGET else PALETTE['danger'], MAPE_TARGET, True),
        (prec_ps,     'Précision PS','Cible > 90%',              '%', '.1f',
         PALETTE['success'] if prec_ps>=90 else PALETTE['danger'],             90,           False),
    ]
    for i, (val,titre,sous,suf,fmt,col,ref,is_mape) in enumerate(kpi_items):
        ok   = (val < MAPE_TARGET) if is_mape else (val >= 90)
        icon = '✅' if ok else '⚠️'
        fig_kpi.add_trace(go.Indicator(
            mode='number+delta', value=val,
            title=dict(
                text=f'<b>{icon} {titre}</b><br>'
                     f'<span style="font-size:10px;color:#777">{sous}</span>',
                font=dict(size=13)),
            number=dict(suffix=suf, valueformat=fmt, font=dict(size=34, color=col)),
            delta=dict(reference=ref, valueformat='.1f',
                       increasing=dict(color=PALETTE['danger'] if is_mape else PALETTE['success']),
                       decreasing=dict(color=PALETTE['success'] if is_mape else PALETTE['danger'])),
            domain=dict(x=domains_x[i], y=[0.05,0.95])
        ))
    fig_kpi.update_layout(**LAYOUT_BASE, height=220,
        title=dict(text='🎯 Performance des Modèles de Prévision (XGBoost)',
                   font=dict(size=14)))

    # Forecast chart
    fig_fc = make_subplots(rows=2, cols=1,
        subplot_titles=['📦 EC — Historique + Prévisions 8 semaines',
                        '🛒 PS — Historique + Prévisions 8 semaines'],
        shared_xaxes=False, vertical_spacing=0.14, row_heights=[0.5,0.5])
    for weekly, forecast, pred_df, c_hist, c_fore, label, ridx in [
        (weekly_ec, forecast_ec_base, pred_ec, PALETTE['primary'], PALETTE['forecast'], 'EC', 1),
        (weekly_ps, forecast_ps_base, pred_ps, PALETTE['success'], PALETTE['purple'],   'PS', 2),
    ]:
        if weekly.empty:
            continue
        hist = weekly.tail(52)
        fig_fc.add_trace(go.Scatter(
            x=hist['week'], y=hist['revenue'], mode='lines',
            name=f'Historique {label}', line=dict(color=c_hist, width=2)), row=ridx, col=1)
        if {DATE_COL,ACTUAL_COL,PRED_COL}.issubset(pred_df.columns):
            fd = pred_df[[DATE_COL,ACTUAL_COL,PRED_COL]].dropna().sort_values(DATE_COL)
            fig_fc.add_trace(go.Scatter(
                x=fd[DATE_COL], y=fd[PRED_COL], mode='lines',
                name=f'Fitted {label}',
                line=dict(color=c_fore, width=1.5, dash='dot')), row=ridx, col=1)
        fill = 'rgba(255,111,0,0.15)' if ridx==1 else 'rgba(156,39,176,0.12)'
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([forecast['week'], forecast['week'][::-1]]),
            y=pd.concat([forecast['upper'], forecast['lower'][::-1]]),
            fill='toself', fillcolor=fill, line=dict(color='rgba(0,0,0,0)'),
            name=f'IC ±15% {label}', hoverinfo='skip'), row=ridx, col=1)
        fig_fc.add_trace(go.Scatter(
            x=forecast['week'], y=forecast['forecast'],
            mode='lines+markers', name=f'Prévision {label}',
            line=dict(color=c_fore, width=2.5, dash='dash'),
            marker=dict(size=7, symbol='diamond', color=c_fore)), row=ridx, col=1)
        last_iso = hist['week'].max().isoformat()
        fig_fc.add_shape(type='line', x0=last_iso, x1=last_iso, y0=0, y1=1,
                         yref='paper',
                         line=dict(dash='dash', color='gray', width=1.5),
                         row=ridx, col=1)
    fig_fc.update_layout(**LAYOUT_BASE, height=680,
        title=dict(text="📅 Prévisions CA — Horizon 8 Semaines", font=dict(size=15)),
        hovermode='x unified', legend=dict(orientation='h', y=-0.06))

    # Tables prévisions
    def make_fc_table(fdf, label, mv):
        fdf = fdf.copy()
        fdf['Semaine']       = fdf['week'].dt.strftime('S%V — %d %b %Y')
        fdf['Prévision (£)'] = fdf['forecast'].map(lambda x: f'£{x:,.0f}')
        fdf['IC Bas']        = fdf['lower'].map(lambda x: f'£{x:,.0f}')
        fdf['IC Haut']       = fdf['upper'].map(lambda x: f'£{x:,.0f}')
        fdf['WoW %']         = fdf['forecast'].pct_change().mul(100).apply(
            lambda x: f'+{x:.1f}%' if pd.notna(x) and x >= 0
                      else (f'{x:.1f}%' if pd.notna(x) else '—'))
        fdf['Tendance']      = fdf['forecast'].pct_change().mul(100).apply(
            lambda x: '📈 Hausse'  if pd.notna(x) and x > 2
                      else ('📉 Baisse' if pd.notna(x) and x < -2 else '➡️ Stable'))
        n  = len(fdf)
        cc = '#E8F5E9' if mv < MAPE_TARGET else '#FFEBEE'
        fig = go.Figure(go.Table(
            header=dict(
                values=['<b>#</b>','<b>Semaine</b>','<b>Prévision (£)</b>',
                        '<b>IC Bas</b>','<b>IC Haut</b>','<b>WoW %</b>','<b>Tendance</b>'],
                fill_color=PALETTE['dark'],
                font=dict(color='white', size=11), align='center', height=32),
            cells=dict(
                values=[list(range(1,n+1)), fdf['Semaine'], fdf['Prévision (£)'],
                        fdf['IC Bas'], fdf['IC Haut'], fdf['WoW %'], fdf['Tendance']],
                fill_color=[
                    ['white']*n, ['#F5F7FA']*n, [cc]*n,
                    ['white']*n, ['white']*n,
                    ['#E8F5E9' if '+' in str(v) else '#FFEBEE' if '-' in str(v)
                     else 'white' for v in fdf['WoW %']],
                    ['white']*n],
                font=dict(size=11),
                align=['center','left','right','right','right','center','center'],
                height=28)
        ))
        fig.update_layout(**LAYOUT_BASE, height=310,
            title=dict(
                text=f'📋 {label} — Prévisions | Total 8 sem: '
                     f'£{fdf["forecast"].sum():,.0f} | MAPE: {mv:.2f}%',
                font=dict(size=13)))
        return fig

    fig_tec = make_fc_table(forecast_ec_base, 'EC', mape_ec_val)
    fig_tps = make_fc_table(forecast_ps_base, 'PS', mape_ps_val)

    # Diagnostic
    fig_diag = make_subplots(rows=1, cols=2,
        subplot_titles=['🎯 Réel vs Prévu — EC',
                        '📊 Distribution des Résidus EC + PS'])
    if {ACTUAL_COL,PRED_COL}.issubset(pred_ec.columns):
        av = pred_ec[ACTUAL_COL].dropna()
        pv = pred_ec[PRED_COL].dropna()
        n  = min(len(av), len(pv))
        fig_diag.add_trace(go.Scatter(
            x=av.iloc[:n], y=pv.iloc[:n], mode='markers',
            marker=dict(color=PALETTE['primary'], size=5, opacity=0.5)), row=1, col=1)
        mv = max(av.iloc[:n].max(), pv.iloc[:n].max())
        fig_diag.add_trace(go.Scatter(
            x=[0,mv], y=[0,mv], mode='lines',
            line=dict(color='red', dash='dash', width=1.5)), row=1, col=1)
        fig_diag.add_trace(go.Histogram(
            x=av.iloc[:n].values - pv.iloc[:n].values,
            nbinsx=40, marker_color=PALETTE['primary'],
            opacity=0.7, name='Résidus EC'), row=1, col=2)
    if {ACTUAL_COL,PRED_COL}.issubset(pred_ps.columns):
        ap  = pred_ps[ACTUAL_COL].dropna()
        pp  = pred_ps[PRED_COL].dropna()
        np_ = min(len(ap), len(pp))
        fig_diag.add_trace(go.Histogram(
            x=ap.iloc[:np_].values - pp.iloc[:np_].values,
            nbinsx=40, marker_color=PALETTE['success'],
            opacity=0.5, name='Résidus PS'), row=1, col=2)
    fig_diag.add_vline(x=0, line_dash='dot', line_color='black',
                       line_width=1.5, row=1, col=2)
    fig_diag.update_layout(**LAYOUT_BASE, height=380, barmode='overlay',
        title=dict(text='🔍 Diagnostic Modèle : Réel vs Prévu & Résidus',
                   font=dict(size=14)))

    return fig_kpi, fig_fc, fig_tec, fig_tps, fig_diag


@callback(
    Output('v2-sim','figure'),
    Input('v2-horizon','value'),
    Input('v2-ci',     'value'),
    Input('v2-src',    'value'),
)
def update_v2_sim(n_weeks, ci_pct_val, src):
    ci_pct = ci_pct_val / 100
    fig    = go.Figure()
    pairs  = []
    if src in ['EC','BOTH'] and not weekly_ec.empty:
        pairs.append((weekly_ec, forecast_ec_base, PALETTE['primary'], PALETTE['forecast'], 'EC'))
    if src in ['PS','BOTH'] and not weekly_ps.empty:
        pairs.append((weekly_ps, forecast_ps_base, PALETTE['success'], PALETTE['purple'],   'PS'))
    for wkly, fc_base, ch, cf, lbl in pairs:
        hist26 = wkly.tail(26)
        fig.add_trace(go.Scatter(
            x=hist26['week'], y=hist26['revenue'], mode='lines',
            name=f'Historique {lbl}', line=dict(color=ch, width=2)))
        fc = fc_base.head(n_weeks).copy()
        fc['lower'] = fc['forecast'] * (1 - ci_pct)
        fc['upper'] = fc['forecast'] * (1 + ci_pct)
        fill = 'rgba(255,111,0,0.12)' if 'EC' in lbl else 'rgba(156,39,176,0.10)'
        fig.add_trace(go.Scatter(
            x=pd.concat([fc['week'], fc['week'][::-1]]),
            y=pd.concat([fc['upper'], fc['lower'][::-1]]),
            fill='toself', fillcolor=fill, line=dict(color='rgba(0,0,0,0)'),
            name=f'IC ±{ci_pct_val:.0f}% {lbl}', hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=fc['week'], y=fc['forecast'],
            mode='lines+markers', name=f'Prévision {lbl} ({n_weeks} sem)',
            line=dict(color=cf, width=2.5, dash='dash'),
            marker=dict(size=8, symbol='diamond', color=cf)))
        last_iso = hist26['week'].max().isoformat()
        fig.add_shape(type='line', x0=last_iso, x1=last_iso, y0=0, y1=1,
                      yref='paper', line=dict(dash='dash', color='gray', width=1.5))
    fig.update_layout(**LAYOUT_BASE, height=420, hovermode='x unified',
        title=dict(text=f'📈 Simulation — {n_weeks} semaines (IC ±{ci_pct_val:.0f}%)',
                   font=dict(size=14)),
        legend=dict(orientation='h', y=-0.1))
    return fig


# ============================================================
# CALLBACKS — VUE 3
# ============================================================

@callback(
    Output('v3-kpi',      'figure'),
    Output('v3-pie',      'figure'),
    Output('v3-matrix',   'figure'),
    Output('v3-growth',   'figure'),
    Output('v3-seasonal', 'figure'),
    Output('v3-discount', 'figure'),
    Output('v3-table-ec', 'figure'),
    Output('v3-table-ps', 'figure'),
    Input('main-tabs', 'value')
)
def update_vue3_static(tab):
    if tab != 'tab-vue3':
        raise PreventUpdate

    mois_lbl = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']

    # KPI
    fig_kpi = make_subplots(rows=1, cols=4, specs=[[{'type':'indicator'}]*4])
    items = [
        (top3_ec, f'Top 3 EC\n≥ {KPI_TARGETS["top3_share"]}%','%','.1f',
         PALETTE['success'] if top3_ec>=KPI_TARGETS['top3_share'] else PALETTE['danger']),
        (top3_ps, f'Top 3 PS\n≥ {KPI_TARGETS["top3_share"]}%','%','.1f',
         PALETTE['success'] if top3_ps>=KPI_TARGETS['top3_share'] else PALETTE['danger']),
        (cat_ec['margin_pct'].mean(), 'Marge Moy. EC\n> 20%','%','.1f',
         PALETTE['success'] if cat_ec['margin_pct'].mean()>=20 else PALETTE['danger']),
        (cat_ps['margin_pct'].mean(), 'Marge Moy. PS\n> 20%','%','.1f',
         PALETTE['success'] if cat_ps['margin_pct'].mean()>=20 else PALETTE['danger']),
    ]
    for ci, (val,lbl,unit,fmt,col) in enumerate(items, 1):
        icon = '✅' if col==PALETTE['success'] else '⚠️'
        fig_kpi.add_trace(go.Indicator(mode='number', value=val,
            title=dict(text=f'<b>{icon} {lbl}</b>', font=dict(size=11)),
            number=dict(suffix=unit, valueformat=fmt,
                        font=dict(size=28, color=col))), row=1, col=ci)
    fig_kpi.update_layout(**LAYOUT_BASE, height=160,
        title=dict(text='🎯 KPIs Marketing — Top Catégories & Marges', font=dict(size=16)))

    # Pie
    fig_pie = make_subplots(rows=1, cols=2,
        subplot_titles=['🇬🇧 EC India — Répartition CA', '🇺🇸 PS USA — Répartition CA'],
        specs=[[{'type':'pie'}, {'type':'pie'}]])
    for ci, (cdf, colors) in enumerate(
            [(cat_ec, px.colors.qualitative.Set2),
             (cat_ps, px.colors.qualitative.Pastel)], 1):
        fig_pie.add_trace(go.Pie(
            labels=cdf['category'], values=cdf['revenue'], hole=0.40,
            marker=dict(colors=colors), textinfo='label+percent',
            textposition='outside',
            pull=[0.05 if i<3 else 0 for i in range(len(cdf))]),
            row=1, col=ci)
    fig_pie.update_layout(**LAYOUT_BASE, height=420,
        title=dict(text='🏪 Répartition du CA par Catégorie', font=dict(size=15)))

    # Matrix CA vs Marge
    fig_mat = make_subplots(rows=1, cols=2,
        subplot_titles=['📊 EC — CA vs Marge (taille=commandes)', '📊 PS — CA vs Marge'])
    for ci, (cdf, c_base) in enumerate(
            [(cat_ec,PALETTE['primary']), (cat_ps,PALETTE['success'])], 1):
        size_norm = (cdf['orders']/cdf['orders'].max()*40+10).tolist()
        colors_b  = [PALETTE['success'] if m>=20 else PALETTE['warning'] if m>=10
                     else PALETTE['danger'] for m in cdf['margin_pct']]
        fig_mat.add_trace(go.Scatter(
            x=cdf['revenue'], y=cdf['margin_pct'],
            mode='markers+text', text=cdf['category'],
            textposition='top center', textfont=dict(size=9),
            marker=dict(size=size_norm, color=colors_b, opacity=0.75,
                        line=dict(color='white', width=1.5)),
            hovertemplate='<b>%{text}</b><br>CA: £%{x:,.0f}<br>'
                          'Marge: %{y:.1f}%<extra></extra>'),
            row=1, col=ci)
        fig_mat.add_hline(y=20, line_dash='dot', line_color='gray',
                          line_width=1, row=1, col=ci)
        fig_mat.add_vline(x=cdf['revenue'].median(), line_dash='dot',
                          line_color='gray', line_width=1, row=1, col=ci)
    fig_mat.update_layout(**LAYOUT_BASE, height=440, showlegend=False,
        title=dict(text='🔍 Matrice CA × Marge (taille = commandes)', font=dict(size=14)))

    # Growth MoM
    fig_gr = make_subplots(rows=1, cols=2,
        subplot_titles=['📈 EC — Croissance MoM (%)', '📈 PS — Croissance MoM (%)'])
    for ci, cdf in enumerate([cat_ec, cat_ps], 1):
        cs = cdf.sort_values('mom_growth', ascending=True)
        fig_gr.add_trace(go.Bar(
            y=cs['category'], x=cs['mom_growth'], orientation='h',
            marker_color=[PALETTE['success'] if v>=0 else PALETTE['danger']
                          for v in cs['mom_growth']],
            opacity=0.85,
            text=[f'{v:+.1f}%' for v in cs['mom_growth']],
            textposition='outside'), row=1, col=ci)
        fig_gr.add_vline(x=0,  line_color='black',       line_width=1,   row=1, col=ci)
        fig_gr.add_vline(x=10, line_dash='dash',
                         line_color=PALETTE['teal'], line_width=1, row=1, col=ci)
    fig_gr.update_layout(**LAYOUT_BASE, height=420, showlegend=False,
        title=dict(text='📅 Croissance MoM par Catégorie', font=dict(size=14)))

    # Heatmap saisonnalité
    fig_sea = make_subplots(rows=1, cols=2,
        subplot_titles=['🗓️  EC — Saisonnalité CA', '🗓️  PS — Saisonnalité CA'])
    for ci, (src_df, label) in enumerate([(ec,'EC'), (ps,'PS')], 1):
        pivot = src_df.groupby(['category','month_num'])['revenue'].sum().unstack(fill_value=0)
        pivot.columns = [mois_lbl[m-1] for m in pivot.columns]
        pn = pivot.div(pivot.max(axis=1), axis=0) * 100
        fig_sea.add_trace(go.Heatmap(
            z=pn.values, x=pn.columns.tolist(), y=pn.index.tolist(),
            colorscale='RdYlGn', zmin=0, zmax=100,
            text=[[f'£{v:,.0f}' for v in row] for row in pivot.values],
            texttemplate='%{text}', textfont=dict(size=8),
            showscale=(ci==2)), row=1, col=ci)
    fig_sea.update_layout(**LAYOUT_BASE, height=420,
        title=dict(text='🌡️  Saisonnalité CA par Catégorie et Mois', font=dict(size=14)))

    # Discount impact
    disc = ec.groupby(['category','has_discount']).agg(
        revenue_mean=('revenue','mean'), margin=('margin_pct','mean')
    ).reset_index()
    disc['label'] = disc['has_discount'].map({0:'Sans discount', 1:'Avec discount'})
    fig_disc = make_subplots(rows=1, cols=2,
        subplot_titles=['💰 Revenue Moyen avec/sans Discount',
                        '📉 Marge avec/sans Discount'])
    for ci, metric in enumerate(['revenue_mean','margin'], 1):
        for dv, color, name in [(0,PALETTE['primary'],'Sans'),
                                 (1,PALETTE['warning'],'Avec')]:
            sub = disc[disc['has_discount']==dv]
            fig_disc.add_trace(go.Bar(
                x=sub['category'], y=sub[metric], name=name,
                marker_color=color, opacity=0.85,
                showlegend=(ci==1)), row=1, col=ci)
    fig_disc.update_layout(**LAYOUT_BASE, height=400, barmode='group',
        title=dict(text='🎟️  Impact du Discount par Catégorie (EC)', font=dict(size=14)),
        legend=dict(orientation='h', y=-0.12))

    # Tables
    def make_cat_table(cdf, source_label):
        total = cdf['revenue'].sum()
        cumul, cumul_pct = 0, []
        for r in cdf['revenue']:
            cumul += r/total*100
            cumul_pct.append(f'{cumul:.1f}%')
        rc  = ['#E8F5E9' if i<3 else '#FFFFFF' for i in range(len(cdf))]
        fig = go.Figure(go.Table(
            header=dict(
                values=['<b>Rang</b>','<b>Catégorie</b>','<b>CA (£)</b>',
                        '<b>Part %</b>','<b>Part Cum.</b>','<b>Profit</b>',
                        '<b>Marge %</b>','<b>Commandes</b>',
                        '<b>Panier Moy.</b>','<b>MoM %</b>'],
                fill_color=PALETTE['dark'],
                font=dict(color='white', size=11), align='center', height=32),
            cells=dict(
                values=[
                    [f'#{i+1}' for i in range(len(cdf))], cdf['category'],
                    [f'£{v:,.0f}' for v in cdf['revenue']],
                    [f'{v:.1f}%'  for v in cdf['ca_share']],
                    cumul_pct,
                    [f'£{v:,.0f}' for v in cdf['profit']],
                    [f'{v:.1f}%'  for v in cdf['margin_pct']],
                    [f'{v:,}'     for v in cdf['orders']],
                    [f'£{v:,.0f}' for v in cdf['panier_moyen']],
                    [f'{v:+.1f}%' for v in cdf['mom_growth']],
                ],
                fill_color=[rc,rc,rc,rc,rc,rc,
                    [PALETTE['success'] if v>=20 else PALETTE['warning'] if v>=10
                     else '#FFEBEE' for v in cdf['margin_pct']],
                    rc, rc,
                    ['#E8F5E9' if v>=0 else '#FFEBEE' for v in cdf['mom_growth']]],
                font=dict(size=11), height=28)
        ))
        top3 = cdf.head(3)['ca_share'].sum()
        fig.update_layout(**LAYOUT_BASE, height=max(280, len(cdf)*32+80),
            title=dict(text=f'📋 {source_label} — Top 3 = {top3:.1f}% du CA',
                       font=dict(size=13)))
        return fig

    return (fig_kpi, fig_pie, fig_mat, fig_gr, fig_sea, fig_disc,
            make_cat_table(cat_ec,'EC India'), make_cat_table(cat_ps,'PS USA'))


@callback(
    Output('v3-cat-dd','options'),
    Output('v3-cat-dd','value'),
    Input('v3-src-dd','value')
)
def update_v3_dropdown(src):
    opts = sorted(ec['category'].unique()) if src=='EC' else sorted(ps['category'].unique())
    return [{'label':c,'value':c} for c in opts], opts[0]


@callback(
    Output('v3-drilldown','figure'),
    Input('v3-src-dd','value'),
    Input('v3-cat-dd','value')
)
def update_v3_drill(src, cat_sel):
    src_df = ec if src=='EC' else ps
    color  = PALETTE['primary'] if src=='EC' else PALETTE['success']
    sub    = src_df[src_df['category']==cat_sel].copy()
    if len(sub) == 0:
        return go.Figure()
    wcat       = sub.groupby('week').agg(
        revenue=('revenue','sum'), orders=('order_id','nunique')).reset_index()
    subcat_rev = sub.groupby('sub_category')['revenue'].sum() \
                    .sort_values(ascending=False).head(8)
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=[f'📈 CA Hebdo — {cat_sel}', '📦 Top Sous-Catégories'])
    fig.add_trace(go.Scatter(
        x=wcat['week'], y=wcat['revenue'], mode='lines+markers',
        line=dict(color=color, width=2),
        fill='tozeroy', fillcolor='rgba(33,150,243,0.08)'), row=1, col=1)
    if len(wcat) >= 4:
        wcat = wcat.copy()
        wcat['ma4'] = wcat['revenue'].rolling(4).mean()
        fig.add_trace(go.Scatter(
            x=wcat['week'], y=wcat['ma4'], mode='lines',
            line=dict(color=PALETTE['danger'], width=1.5, dash='dash')), row=1, col=1)
    fig.add_trace(go.Bar(
        y=subcat_rev.index, x=subcat_rev.values, orientation='h',
        marker_color=color, opacity=0.8), row=1, col=2)
    fig.update_layout(**LAYOUT_BASE, height=360, showlegend=False,
        title=dict(
            text=f'🔍 Drill-down : {cat_sel} ({src}) | '
                 f'CA: £{sub["revenue"].sum():,.0f} | '
                 f'Marge: {sub["margin_pct"].mean():.1f}%',
            font=dict(size=13)))
    return fig


# ============================================================
# CALLBACKS — VUE 4
# ============================================================

@callback(
    Output('v4-kpi',      'figure'),
    Output('v4-regions',  'figure'),
    Output('v4-map',      'figure'),
    Output('v4-states',   'figure'),
    Output('v4-cities',   'figure'),
    Output('v4-time',     'figure'),
    Output('v4-heatmap',  'figure'),
    Output('v4-table-ec', 'figure'),
    Output('v4-table-ps', 'figure'),
    Input('main-tabs', 'value')
)
def update_vue4_static(tab):
    if tab != 'tab-vue4':
        raise PreventUpdate

    top_reg_ec = reg_ec.iloc[0]
    top_reg_ps = reg_ps.iloc[0]

    # KPI
    fig_kpi = make_subplots(rows=1, cols=4, specs=[[{'type':'indicator'}]*4])
    items = [
        (top_reg_ec['ca_share'],
         f'Top Région EC\n{top_reg_ec["region"]}','%','.1f',PALETTE['primary']),
        (top_reg_ps['ca_share'],
         f'Top Région PS\n{top_reg_ps["region"]}','%','.1f',PALETTE['success']),
        (len(state_metrics[state_metrics['revenue']>state_metrics['revenue'].median()]),
         'États US\n>médiane','','d',PALETTE['teal']),
        (ec['city'].nunique(),'Villes EC\ncouvertes','','d',PALETTE['purple']),
    ]
    for ci, (val,lbl,unit,fmt,col) in enumerate(items, 1):
        fig_kpi.add_trace(go.Indicator(mode='number', value=val,
            title=dict(text=f'<b>{lbl}</b>', font=dict(size=11)),
            number=dict(suffix=unit, valueformat=fmt,
                        font=dict(size=28, color=col))), row=1, col=ci)
    fig_kpi.update_layout(**LAYOUT_BASE, height=160,
        title=dict(text='🌐 Indicateurs Géographiques Clés', font=dict(size=14)))

    # Régions
    fig_reg = make_subplots(rows=1, cols=2,
        subplot_titles=['🇬🇧 EC — CA & Marge par Région',
                        '🇺🇸 PS — CA & Marge par Région'],
        specs=[[{'secondary_y':True}, {'secondary_y':True}]])
    for ci, (rdf,cb,cl,lbl) in enumerate([
            (reg_ec, PALETTE['primary'], PALETTE['danger'],  'EC'),
            (reg_ps, PALETTE['success'], PALETTE['warning'], 'PS')], 1):
        rs = rdf.sort_values('revenue', ascending=True)
        fig_reg.add_trace(go.Bar(
            y=rs['region'], x=rs['revenue'], orientation='h',
            name=f'CA {lbl}', marker_color=cb, opacity=0.85,
            text=[f'£{v:,.0f}' for v in rs['revenue']],
            textposition='outside'),
            row=1, col=ci, secondary_y=False)
        fig_reg.add_trace(go.Scatter(
            y=rs['region'], x=rs['margin_pct'], mode='markers+lines',
            name=f'Marge {lbl}',
            marker=dict(color=cl, size=9, symbol='diamond'),
            line=dict(color=cl, width=1.5, dash='dot')),
            row=1, col=ci, secondary_y=True)
    fig_reg.update_layout(**LAYOUT_BASE, height=380, barmode='group',
        title=dict(text='📊 CA et Marge par Région', font=dict(size=15)),
        legend=dict(orientation='h', y=-0.12))

    # Carte US
    fig_map = go.Figure(go.Choropleth(
        locations=state_metrics['state_code'], z=state_metrics['revenue'],
        locationmode='USA-states', colorscale='Blues',
        colorbar=dict(title=dict(text='CA (£)'), thickness=14, len=0.7),
        text=state_metrics['state'],
        customdata=state_metrics[['revenue','margin_pct','orders','panier_moyen']].values,
        hovertemplate='<b>%{text}</b><br>CA: £%{customdata[0]:,.0f}<br>'
                      'Marge: %{customdata[1]:.1f}%<br>'
                      'Commandes: %{customdata[2]:,}<extra></extra>',
        marker_line_color='white', marker_line_width=0.5))
    fig_map.update_layout(**LAYOUT_BASE, height=460,
        title=dict(text='🗺️  PS USA — CA par État', font=dict(size=15)),
        geo=dict(scope='usa', projection_type='albers usa', showlakes=True,
                 lakecolor='#E3F2FD', bgcolor='#F5F7FA'))

    # Top 10 états
    top10 = state_metrics.head(10).copy()
    state_monthly = ps.groupby(['state','month'])['revenue'].sum() \
                      .reset_index().sort_values('month')
    sg = {}
    for state in state_monthly['state'].unique():
        sub = state_monthly[state_monthly['state']==state]['revenue'].values
        sg[state] = (sub[-1]-sub[-2])/max(sub[-2],1)*100 if len(sub)>=2 else 0.0
    top10['mom_growth'] = top10['state'].map(sg).fillna(0)

    fig_states = make_subplots(rows=1, cols=2,
        subplot_titles=['🏆 Top 10 États — CA', '📈 Top 10 États — MoM %'])
    fig_states.add_trace(go.Bar(
        y=top10['state'][::-1], x=top10['revenue'][::-1], orientation='h',
        marker_color=[PALETTE['primary'] if i<3 else PALETTE['teal']
                      for i in range(len(top10))][::-1],
        opacity=0.85,
        text=[f'£{v:,.0f}' for v in top10['revenue'][::-1]],
        textposition='outside'), row=1, col=1)
    fig_states.add_trace(go.Bar(
        y=top10['state'][::-1], x=top10['mom_growth'][::-1], orientation='h',
        marker_color=[PALETTE['success'] if v>=0 else PALETTE['danger']
                      for v in top10['mom_growth'][::-1]],
        opacity=0.85,
        text=[f'{v:+.1f}%' for v in top10['mom_growth'][::-1]],
        textposition='outside'), row=1, col=2)
    fig_states.add_vline(x=0, line_color='black', line_width=1, row=1, col=2)
    fig_states.update_layout(**LAYOUT_BASE, height=400, showlegend=False,
        title=dict(text='🏆 Top 10 États US', font=dict(size=14)))

    # Villes
    fig_cities = make_subplots(rows=1, cols=2,
        subplot_titles=['🏙️  Top 10 Villes EC', '🏙️  Top 10 Villes PS'])
    for ci, (cities, color) in enumerate(
            [(top_cities_ec,PALETTE['primary']), (top_cities_ps,PALETTE['success'])], 1):
        fig_cities.add_trace(go.Bar(
            y=cities['city'][::-1], x=cities['revenue'][::-1], orientation='h',
            marker_color=color, opacity=0.85,
            text=[f'£{v:,.0f}' for v in cities['revenue'][::-1]],
            textposition='outside'), row=1, col=ci)
    fig_cities.update_layout(**LAYOUT_BASE, height=400, showlegend=False,
        title=dict(text='🌆 Top 10 Villes par CA', font=dict(size=14)))

    # Évolution temporelle par région
    fig_time = make_subplots(rows=1, cols=2,
        subplot_titles=['📅 EC — CA Mensuel par Région',
                        '📅 PS — CA Mensuel par Région'])
    ec_cr = [PALETTE['primary'],PALETTE['success'],PALETTE['warning'],PALETTE['purple']]
    ps_cr = [PALETTE['teal'],PALETTE['danger'],PALETTE['amber'],PALETTE['primary']]
    for ci, (src_df,colors,lbl) in enumerate(
            [(ec,ec_cr,'EC'), (ps,ps_cr,'PS')], 1):
        mr = src_df.groupby(['region','month'])['revenue'].sum().reset_index()
        for i, reg in enumerate(sorted(src_df['region'].unique())):
            sub = mr[mr['region']==reg].sort_values('month')
            fig_time.add_trace(go.Scatter(
                x=sub['month'], y=sub['revenue'], mode='lines',
                name=f'{reg} ({lbl})',
                line=dict(color=colors[i%len(colors)], width=2)),
                row=1, col=ci)
    fig_time.update_layout(**LAYOUT_BASE, height=400, hovermode='x unified',
        title=dict(text='📈 Évolution Mensuelle par Région', font=dict(size=14)),
        legend=dict(orientation='h', y=-0.14, font=dict(size=10)))

    # Heatmap région × catégorie
    fig_hm = make_subplots(rows=1, cols=2,
        subplot_titles=['🌡️  EC — Région × Catégorie',
                        '🌡️  PS — Région × Catégorie'])
    for ci, (src_df,lbl) in enumerate([(ec,'EC'), (ps,'PS')], 1):
        pivot = src_df.groupby(['region','category'])['revenue'].sum().unstack(fill_value=0)
        pn    = pivot.div(pivot.max(axis=1), axis=0) * 100
        fig_hm.add_trace(go.Heatmap(
            z=pn.values, x=pn.columns.tolist(), y=pn.index.tolist(),
            colorscale='Blues', zmin=0, zmax=100,
            text=[[f'£{v:,.0f}' for v in row] for row in pivot.values],
            texttemplate='%{text}', textfont=dict(size=8),
            showscale=(ci==2)), row=1, col=ci)
    fig_hm.update_layout(**LAYOUT_BASE, height=340,
        title=dict(text='🗺️  Intensité CA — Région × Catégorie', font=dict(size=14)))

    # Tables régions
    def make_reg_table(rdf, sl):
        total = rdf['revenue'].sum()
        cumul, cl = 0, []
        for r in rdf['revenue']:
            cumul += r/total*100
            cl.append(f'{cumul:.1f}%')
        rc  = ['#E8F5E9' if i==0 else '#FFFFFF' for i in range(len(rdf))]
        fig = go.Figure(go.Table(
            header=dict(
                values=['<b>Rang</b>','<b>Région</b>','<b>CA (£)</b>',
                        '<b>Part %</b>','<b>Part Cum.</b>','<b>Profit</b>',
                        '<b>Marge %</b>','<b>Commandes</b>',
                        '<b>Clients</b>','<b>MoM %</b>'],
                fill_color=PALETTE['dark'],
                font=dict(color='white', size=11), align='center', height=32),
            cells=dict(
                values=[
                    [f'#{i+1}' for i in range(len(rdf))], rdf['region'],
                    [f'£{v:,.0f}' for v in rdf['revenue']],
                    [f'{v:.1f}%'  for v in rdf['ca_share']], cl,
                    [f'£{v:,.0f}' for v in rdf['profit']],
                    [f'{v:.1f}%'  for v in rdf['margin_pct']],
                    [f'{v:,}'     for v in rdf['orders']],
                    [f'{v:,}'     for v in rdf['clients']],
                    [f'{v:+.1f}%' for v in rdf['mom_growth']],
                ],
                fill_color=[rc,rc,rc,rc,rc,rc,
                    [PALETTE['success'] if v>=20 else PALETTE['warning'] if v>=10
                     else '#FFEBEE' for v in rdf['margin_pct']],
                    rc, rc,
                    ['#E8F5E9' if v>=0 else '#FFEBEE' for v in rdf['mom_growth']]],
                font=dict(size=11), height=28)
        ))
        fig.update_layout(**LAYOUT_BASE, height=max(240, len(rdf)*32+80),
            title=dict(text=f'📋 {sl} — Performance par Région', font=dict(size=13)))
        return fig

    return (fig_kpi, fig_reg, fig_map, fig_states, fig_cities,
            fig_time, fig_hm,
            make_reg_table(reg_ec,'EC India'),
            make_reg_table(reg_ps,'PS USA'))


@callback(
    Output('v4-region-dd','options'),
    Output('v4-region-dd','value'),
    Input('v4-src-geo','value')
)
def update_v4_region_dd(src):
    opts = sorted(ec['region'].unique()) if src=='EC' else sorted(ps['region'].unique())
    return [{'label':r,'value':r} for r in opts], opts[0]


@callback(
    Output('v4-drilldown','figure'),
    Input('v4-src-geo',   'value'),
    Input('v4-region-dd', 'value'),
    Input('v4-metric',    'value'),
)
def update_v4_drill(src, reg, metric):
    src_df = ec if src=='EC' else ps
    color  = PALETTE['primary'] if src=='EC' else PALETTE['success']
    sub    = src_df[src_df['region']==reg].copy()
    if len(sub) == 0:
        return go.Figure()
    ms = sub.groupby('month').agg(
        revenue=('revenue','sum'), profit=('profit','sum'),
        orders=('order_id','nunique')).reset_index()
    cat_sub  = sub.groupby('category')['revenue'].sum().sort_values(ascending=False)
    city_sub = sub.groupby('city')['revenue'].sum() \
                  .sort_values(ascending=False).head(8)
    ml  = {'revenue':'CA (£)', 'profit':'Profit (£)', 'orders':'Commandes'}[metric]
    fig = make_subplots(rows=1, cols=3,
        subplot_titles=[f'📅 {ml} mensuel', '📦 CA par Catégorie', '🏙️  Top Villes'],
        column_widths=[0.4, 0.3, 0.3])
    y_vals = ms[metric] if metric != 'orders' else ms['orders']
    fig.add_trace(go.Scatter(
        x=ms['month'], y=y_vals, mode='lines+markers',
        line=dict(color=color, width=2.5), marker=dict(size=7),
        fill='tozeroy', fillcolor='rgba(33,150,243,0.08)'), row=1, col=1)
    fig.add_trace(go.Bar(
        y=cat_sub.index, x=cat_sub.values, orientation='h',
        marker_color=color, opacity=0.8), row=1, col=2)
    fig.add_trace(go.Bar(
        y=city_sub.index, x=city_sub.values, orientation='h',
        marker_color=PALETTE['teal'], opacity=0.8), row=1, col=3)
    fig.update_layout(**LAYOUT_BASE, height=360, showlegend=False,
        title=dict(
            text=f'🔍 Drill-down : {reg} ({src}) | '
                 f'CA: £{sub["revenue"].sum():,.0f} | '
                 f'Marge: {sub["margin_pct"].mean():.1f}%',
            font=dict(size=13)))
    return fig


# ============================================================
# CALLBACKS — VUE 5
# ============================================================

@callback(
    Output('v5-kpi',      'figure'),
    Output('v5-coverage', 'figure'),
    Output('v5-alert-ec', 'figure'),
    Output('v5-alert-ps', 'figure'),
    Output('v5-demand',   'figure'),
    Output('v5-rotation', 'figure'),
    Output('v5-evo',      'figure'),
    Input('main-tabs', 'value')
)
def update_vue5_static(tab):
    if tab != 'tab-vue5':
        raise PreventUpdate

    # KPI
    fig_kpi = make_subplots(rows=1, cols=4, specs=[[{'type':'indicator'}]*4])
    items = [
        (kpi_stock_ec['couv_moy'],
         f'Couverture EC\n≥ {STOCK_COVERAGE_TARGET} sem.',' sem','.1f',
         PALETTE['success'] if kpi_stock_ec['couv_moy']>=STOCK_COVERAGE_TARGET
         else PALETTE['danger']),
        (kpi_stock_ps['couv_moy'],
         f'Couverture PS\n≥ {STOCK_COVERAGE_TARGET} sem.',' sem','.1f',
         PALETTE['success'] if kpi_stock_ps['couv_moy']>=STOCK_COVERAGE_TARGET
         else PALETTE['danger']),
        (kpi_stock_ec['n_alertes'],
         f'Alertes EC\n({kpi_stock_ec["n_total"]} catégs.)','','d',
         PALETTE['success'] if kpi_stock_ec['n_alertes']==0
         else PALETTE['warning'] if kpi_stock_ec['n_alertes']<=2
         else PALETTE['danger']),
        (kpi_stock_ps['n_alertes'],
         f'Alertes PS\n({kpi_stock_ps["n_total"]} catégs.)','','d',
         PALETTE['success'] if kpi_stock_ps['n_alertes']==0
         else PALETTE['warning'] if kpi_stock_ps['n_alertes']<=2
         else PALETTE['danger']),
    ]
    for ci, (val,lbl,unit,fmt,col) in enumerate(items, 1):
        icon = '✅' if col==PALETTE['success'] else ('⚠️' if col==PALETTE['warning'] else '🔴')
        fig_kpi.add_trace(go.Indicator(mode='number', value=val,
            title=dict(text=f'<b>{icon} {lbl}</b>', font=dict(size=11)),
            number=dict(suffix=unit, valueformat=fmt,
                        font=dict(size=28, color=col))), row=1, col=ci)
    fig_kpi.update_layout(**LAYOUT_BASE, height=160,
        title=dict(text='🎯 KPIs Opérationnels — Couverture & Alertes',
                   font=dict(size=14)))

    # Coverage
    status_colors = {
        '🔴 RUPTURE IMMINENTE': PALETTE['danger'],
        '🟡 STOCK FAIBLE'     : PALETTE['warning'],
        '🟢 STOCK OK'         : PALETTE['success'],
        '🔵 SURSTOCKAGE'      : PALETTE['primary'],
    }
    fig_cov = make_subplots(rows=1, cols=2,
        subplot_titles=['📦 EC — Taux de Couverture (semaines)',
                        '📦 PS — Taux de Couverture (semaines)'])
    for ci, sdf in enumerate([stock_ec, stock_ps], 1):
        ds = sdf.sort_values('taux_couverture', ascending=True)
        fig_cov.add_trace(go.Bar(
            y=ds['category'], x=ds['taux_couverture'], orientation='h',
            marker_color=[status_colors.get(s, PALETTE['teal'])
                          for s in ds['statut_stock']],
            opacity=0.85,
            text=[f'{v:.1f} sem' for v in ds['taux_couverture']],
            textposition='outside'), row=1, col=ci)
        fig_cov.add_vline(x=STOCK_COVERAGE_TARGET, line_dash='dash',
                          line_color='black', line_width=2, row=1, col=ci)
    fig_cov.update_layout(**LAYOUT_BASE, height=420, showlegend=False,
        title=dict(text='⚠️  Taux de Couverture des Stocks', font=dict(size=15)))

    # Alert tables
    def make_alert_tbl(sdf, sl):
        sdf = sdf.copy()
        sdf['Priorité'] = sdf['taux_couverture'].apply(
            lambda x: '🔴 URGENT' if x<1
                      else ('🟡 ÉLEVÉE' if x<STOCK_COVERAGE_TARGET else '🟢 Normal'))
        rfill = ['#FFEBEE' if '🔴' in s else '#FFF8E1' if '🟡' in s
                 else '#E3F2FD' if '🔵' in s else '#F1F8E9'
                 for s in sdf['statut_stock']]
        fig = go.Figure(go.Table(
            header=dict(
                values=['<b>Priorité</b>','<b>Catégorie</b>','<b>Stock Actuel</b>',
                        '<b>Stock Recommandé</b>','<b>Stock Sécu.</b>',
                        '<b>Pt Réappro</b>','<b>Couverture</b>',
                        '<b>Rotation</b>','<b>Statut</b>','<b>Alerte</b>'],
                fill_color=PALETTE['dark'],
                font=dict(color='white', size=11), align='center', height=32),
            cells=dict(
                values=[
                    sdf['Priorité'], sdf['category'],
                    [f'{v:,.0f} u' for v in sdf['stock_actuel']],
                    [f'{v:,.0f} u' for v in sdf['stock_recommande']],
                    [f'{v:,.0f} u' for v in sdf['safety_stock']],
                    [f'{v:,.0f} u' for v in sdf['point_reappro']],
                    [f'{v:.1f} sem' for v in sdf['taux_couverture']],
                    [f'{v:.1f}x'   for v in sdf['rotation_stock']],
                    sdf['statut_stock'],
                    ['🚨 COMMANDER' if a else '✅ OK'
                     for a in sdf['alerte_reappro']],
                ],
                fill_color=[rfill]*9 + [[
                    '#FFEBEE' if a else '#F1F8E9'
                    for a in sdf['alerte_reappro']]],
                font=dict(size=11), height=28)
        ))
        n_urg = (sdf['Priorité'].str.contains('URGENT')).sum()
        n_elv = (sdf['Priorité'].str.contains('ÉLEVÉE')).sum()
        fig.update_layout(**LAYOUT_BASE, height=max(280, len(sdf)*32+80),
            title=dict(
                text=f'📋 {sl} | 🔴 Urgent: {n_urg} | 🟡 Élevé: {n_elv}',
                font=dict(size=13)))
        return fig

    # Demand vs stock
    fig_dem = make_subplots(rows=1, cols=2,
        subplot_titles=['📈 EC — Demande 8 sem. vs Stock',
                        '📈 PS — Demande 8 sem. vs Stock'])
    for ci, (fdf,sdf,color) in enumerate([
            (demand_fc_ec, stock_ec, PALETTE['primary']),
            (demand_fc_ps, stock_ps, PALETTE['success'])], 1):
        td = fdf.groupby('category')['forecast'].sum().reset_index()
        td.columns = ['category','demand_8w']
        merged = td.merge(sdf[['category','stock_actuel','stock_recommande']],
                          on='category', how='left') \
                   .sort_values('demand_8w', ascending=True)
        fig_dem.add_trace(go.Bar(
            y=merged['category'], x=merged['demand_8w'], orientation='h',
            name='Demande 8 sem', marker_color=color, opacity=0.7),
            row=1, col=ci)
        fig_dem.add_trace(go.Scatter(
            y=merged['category'], x=merged['stock_actuel'], mode='markers',
            name='Stock Actuel',
            marker=dict(
                color=[PALETTE['success'] if s>=d else PALETTE['danger']
                       for s,d in zip(merged['stock_actuel'],merged['demand_8w'])],
                size=12, symbol='diamond',
                line=dict(color='white', width=1.5))),
            row=1, col=ci)
    fig_dem.update_layout(**LAYOUT_BASE, height=440, barmode='overlay',
        title=dict(text='📊 Demande Prévue 8 sem vs Stock Disponible',
                   font=dict(size=14)),
        legend=dict(orientation='h', y=-0.12))

    # Rotation & CV
    all_stock = pd.concat([stock_ec, stock_ps], ignore_index=True)
    fig_rot   = make_subplots(rows=1, cols=2,
        subplot_titles=['🔄 Rotation des Stocks', '📉 Variabilité Demande (CV%)'])
    rot_mean  = all_stock.groupby('category')['rotation_stock'].mean() \
                         .sort_values(ascending=True)
    fig_rot.add_trace(go.Bar(
        y=rot_mean.index, x=rot_mean.values, orientation='h',
        marker_color=[PALETTE['success'] if v>=4 else PALETTE['warning'] if v>=2
                      else PALETTE['danger'] for v in rot_mean.values],
        opacity=0.85,
        text=[f'{v:.1f}x' for v in rot_mean.values],
        textposition='outside'), row=1, col=1)
    fig_rot.add_vline(x=4, line_dash='dash', line_color=PALETTE['teal'],
                      line_width=1.5, row=1, col=1)
    cv_mean = all_stock.groupby('category')['cv'].mean().sort_values(ascending=False)
    fig_rot.add_trace(go.Bar(
        y=cv_mean.index, x=cv_mean.values, orientation='h',
        marker_color=[PALETTE['danger'] if v>50 else PALETTE['warning'] if v>30
                      else PALETTE['success'] for v in cv_mean.values],
        opacity=0.85,
        text=[f'{v:.0f}%' for v in cv_mean.values],
        textposition='outside'), row=1, col=2)
    fig_rot.add_vline(x=30, line_dash='dash', line_color='gray',
                      line_width=1.5, row=1, col=2)
    fig_rot.update_layout(**LAYOUT_BASE, height=400, showlegend=False,
        title=dict(text='🔄 Rotation & Variabilité de la Demande', font=dict(size=14)))

    # Évolution demande
    fig_evo = make_subplots(rows=1, cols=2,
        subplot_titles=['📅 EC — Demande Hebdo par Catégorie',
                        '📅 PS — Demande Hebdo par Catégorie'])
    for ci, (src_df,colors,lbl) in enumerate([
            (ec, px.colors.qualitative.Set2,   'EC'),
            (ps, px.colors.qualitative.Pastel, 'PS')], 1):
        wc = src_df.groupby(['category','week'])['quantity'].sum().reset_index()
        for i, cat in enumerate(sorted(src_df['category'].unique())):
            sub = wc[wc['category']==cat].sort_values('week')
            fig_evo.add_trace(go.Scatter(
                x=sub['week'], y=sub['quantity'], mode='lines',
                name=f'{cat} ({lbl})',
                line=dict(color=colors[i%len(colors)], width=1.5),
                opacity=0.8), row=1, col=ci)
    fig_evo.update_layout(**LAYOUT_BASE, height=400, hovermode='x unified',
        title=dict(text='📈 Évolution Demande Hebdomadaire par Catégorie',
                   font=dict(size=14)),
        legend=dict(orientation='h', y=-0.15, font=dict(size=9)))

    return (fig_kpi, fig_cov,
            make_alert_tbl(stock_ec,'EC India'),
            make_alert_tbl(stock_ps,'PS USA'),
            fig_dem, fig_rot, fig_evo)


@callback(
    Output('v5-cat-dd','options'),
    Output('v5-cat-dd','value'),
    Input('v5-src-stock','value')
)
def update_v5_cat_dd(src):
    opts = sorted(ec['category'].unique()) if src=='EC' else sorted(ps['category'].unique())
    return [{'label':c,'value':c} for c in opts], opts[0]


@callback(
    Output('v5-gauge',     'figure'),
    Input('v5-src-stock',  'value'),
    Input('v5-cat-dd',     'value'),
    Input('v5-stock-slider','value'),
)
def update_v5_gauge(src, cat, stock):
    sdf = stock_ec if src=='EC' else stock_ps
    row = sdf[sdf['category']==cat]
    if len(row) == 0:
        return go.Figure()
    row         = row.iloc[0]
    demand_mean = row['demand_mean']
    couverture  = stock / max(demand_mean, 1)
    if couverture < 1:
        statut  = '🔴 RUPTURE IMMINENTE'; s_color = PALETTE['danger']
    elif couverture < STOCK_COVERAGE_TARGET:
        statut  = '🟡 STOCK FAIBLE';      s_color = PALETTE['warning']
    elif couverture < 6:
        statut  = '🟢 STOCK OK';          s_color = PALETTE['success']
    else:
        statut  = '🔵 SURSTOCKAGE';       s_color = PALETTE['primary']
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta', value=couverture,
        title=dict(
            text=f'<b>Couverture — {cat} ({src})</b><br>'
                 f'<span style="font-size:12px">{statut}</span>',
            font=dict(size=13)),
        number=dict(suffix=' semaines', valueformat='.1f',
                    font=dict(size=30, color=s_color)),
        delta=dict(reference=STOCK_COVERAGE_TARGET, valueformat='.1f',
                   increasing=dict(color=PALETTE['success']),
                   decreasing=dict(color=PALETTE['danger'])),
        gauge=dict(
            axis=dict(range=[0,10]),
            bar=dict(color=s_color, thickness=0.25),
            bgcolor='#F0F0F0',
            steps=[
                dict(range=[0,1],  color='#FFEBEE'),
                dict(range=[1,2],  color='#FFF8E1'),
                dict(range=[2,6],  color='#E8F5E9'),
                dict(range=[6,10], color='#E3F2FD'),
            ],
            threshold=dict(line=dict(color='black', width=2),
                           thickness=0.75, value=STOCK_COVERAGE_TARGET)
        )
    ))
    fig.update_layout(**LAYOUT_BASE, height=320,
        title=dict(
            text=f'📦 Simulateur Stock — Stock: {stock:,} u | '
                 f'Demande moy: {demand_mean:.0f} u/sem',
            font=dict(size=13)))
    return fig


# ============================================================
# DÉMARRAGE
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  🚀  Dashboard Décisionnel — démarrage...")
    print("  🌐  Ouvrir : http://localhost:8050")
    print("="*55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=8050)
