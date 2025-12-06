"""
Dashboard MSAE - Tableau de Bord Intelligent
Mutuelle de Santé des Agents de l'État du Sénégal
Auteurs: Marc MARE & Awa GUEYE
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import time
import threading
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    APSCHEDULER_AVAILABLE = True
except Exception:
    APSCHEDULER_AVAILABLE = False

# ============================================================
# DONNÉES FICTIVES RÉALISTES
# ============================================================

# Données initiales générées dynamiquement


def generate_demo_data(months=12, seed=42):
    np.random.seed(seed)
    # monthly aggregated data
    months_idx = pd.date_range(end=datetime.now(), periods=months, freq='M')
    mois = [d.strftime('%b') for d in months_idx]

    # simulate contributions and claims (in millions)
    cotisations = (42 + np.cumsum(np.random.normal(0.3, 0.8, size=months))).round(1)
    remboursements = (40 + np.cumsum(np.random.normal(0.5, 1.0, size=months))).round(1)

    ratio = (remboursements / cotisations * 100).round(1).tolist()

    ratio_data_local = {'mois': mois, 'ratio': ratio}
    finances_data_local = {'mois': mois, 'cotisations': cotisations.tolist(), 'remboursements': remboursements.tolist()}

    # claim-level synthetic data for fraud detection
    n_claims = 1200
    claim_dates = pd.to_datetime(np.random.choice(months_idx, size=n_claims))
    service_types = np.random.choice(['Médicament', 'Lunettes', 'Analyses', 'Imagerie', 'Hospitalisation'], size=n_claims, p=[0.5,0.15,0.15,0.1,0.1])
    base_amount = {'Médicament': 12000, 'Lunettes': 30000, 'Analyses': 15000, 'Imagerie': 25000, 'Hospitalisation': 80000}
    amounts = []
    providers = []
    processing_days = []
    for s in service_types:
        amt = int(np.random.gamma(2, base_amount[s] / 2))
        # 5% chance of fraud multiplier
        if np.random.rand() < 0.05:
            amt = int(amt * np.random.uniform(2.5, 4.0))
        amounts.append(amt)
        providers.append(f"Prestataire {np.random.randint(1,200):03d}")
        processing_days.append(int(np.random.gamma(3,25)))

    claims = pd.DataFrame({
        'date': claim_dates,
        'service_type': service_types,
        'amount': amounts,
        'provider': providers,
        'processing_days': processing_days
    })

    return ratio_data_local, finances_data_local, claims


# generate once at startup
print("[startup] Génération des données de démonstration... (cela peut prendre quelques secondes)")
ratio_data, finances_data, claims_df = generate_demo_data(months=12)
print(f"[startup] Données générées : {len(ratio_data['mois'])} mois, {len(claims_df)} dossiers");


def recompute_and_cache(seed=None):
    """Régénère les données simulées et recalcule les KPI, stocke dans `cached_kpis`.
    Conçu pour être appelé par un scheduler chaque lundi matin.
    """
    global ratio_data, finances_data, claims_df, cached_kpis
    try:
        s = int(seed) if seed is not None else int(time.time()) % 100000
        rd, fd, cd = generate_demo_data(months=12, seed=s)
        ratio_data = rd
        finances_data = fd
        claims_df = cd
        cached_kpis = compute_kpis_from_simulation(ratio_data, finances_data, claims_df)
        print(f"[scheduler] KPIs recomputed and cached at {datetime.now()}")
    except Exception as e:
        print(f"[scheduler] Error recomputing KPIs: {e}")


def start_weekly_scheduler():
    """Démarre le scheduler pour exécuter `recompute_and_cache` chaque lundi à 06:00."""
    if not APSCHEDULER_AVAILABLE:
        print("[scheduler] APScheduler non disponible — installez 'apscheduler' pour activer le recalcul hebdomadaire.")
        return None
    sched = BackgroundScheduler()
    # Lancer immédiatement une première fois puis cron chaque lundi 06:00
    sched.add_job(recompute_and_cache, 'date', run_date=datetime.now())
    sched.add_job(recompute_and_cache, 'cron', day_of_week='mon', hour=6, minute=0)
    sched.start()
    print("[scheduler] Démarré — recalcul hebdomadaire programmé chaque lundi à 06:00")
    return sched

alertes = [
    {'priorite': 'Critique', 'titre': 'Ratio combiné proche du seuil critique (105%)', 'date': '05/12/2025', 'color': '#dc2626'},
    {'priorite': 'Haute', 'titre': 'Baisse des nouvelles adhésions (-12% ce mois)', 'date': '04/12/2025', 'color': '#ea580c'},
    {'priorite': 'Moyenne', 'titre': '23 prestataires en retard de paiement > 90j', 'date': '03/12/2025', 'color': '#ca8a04'},
    {'priorite': 'Haute', 'titre': 'Pic de remboursements détecté (Pharmacies)', 'date': '02/12/2025', 'color': '#ea580c'},
    {'priorite': 'Basse', 'titre': 'Mise à jour convention clinique Dakar', 'date': '01/12/2025', 'color': '#16a34a'},
]

# ============================================================
# CONFIGURATION APPLICATION
# ============================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
    ],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]
)

app.title = "MSAE - Tableau de Bord"
server = app.server  # Expose server for deployment

# ============================================================
# STYLES CSS
# ============================================================

COLORS = {
    'primary': '#2563eb',
    'primary_dark': '#1e40af',
    'primary_light': '#dbeafe',
    'success': '#10b981',
    'success_light': '#d1fae5',
    'warning': '#f59e0b',
    'warning_light': '#fef3c7',
    'danger': '#ef4444',
    'danger_light': '#fee2e2',
    'info': '#06b6d4',
    'info_light': '#cffafe',
    'purple': '#8b5cf6',
    'purple_light': '#ede9fe',
    'bg_light': '#f8fafc',
    'bg_card': '#ffffff',
    'text_dark': '#0f172a',
    'text_muted': '#64748b',
    'text_light': '#94a3b8',
    'border': '#e2e8f0',
    'border_light': '#f1f5f9'
}

CARD_STYLE = {
    'backgroundColor': COLORS['bg_card'],
    'borderRadius': '16px',
    'padding': '28px',
    'boxShadow': '0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06)',
    'border': f'1px solid {COLORS["border"]}',
    'height': '100%',
    'transition': 'all 0.3s ease'
}

# ============================================================
# COMPOSANTS
# ============================================================

def create_kpi_card(title, value, unit, trend, trend_label, status_color, id_prefix=None, sparkline=None, md=6, lg=3, xl=2):
    """Crée une carte KPI moderne avec gradient et animations"""
    trend_color = COLORS['success'] if trend >= 0 else COLORS['danger']
    trend_icon = "▲" if trend >= 0 else "▼"

    # Couleur de fond gradient basée sur le status_color
    gradient_bg = f'linear-gradient(135deg, {status_color}15 0%, {status_color}08 100%)'

    value_id = f"{id_prefix}-value" if id_prefix else None
    trend_id = f"{id_prefix}-trend" if id_prefix else None

    # special styling for some cards (e.g. top-service long text)
    special_value_style = {}
    special_title_style = {}
    extra_min_height = None
    if id_prefix and 'top-service' in id_prefix:
        special_value_style = {'fontSize': '20px', 'whiteSpace': 'normal', 'wordBreak': 'break-word'}
        special_title_style = {'fontSize': '12px'}
        extra_min_height = '220px'

    card_content = html.Div([
        # Icône décorative en fond
        html.Div(
            style={
                'position': 'absolute',
                'top': '16px',
                'right': '16px',
                'width': '48px',
                'height': '48px',
                'backgroundColor': f'{status_color}20',
                'borderRadius': '12px',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center'
            }
        ),

        # Barre colorée gauche
        html.Div(
            style={
                'width': '5px',
                'height': '80px',
                'backgroundColor': status_color,
                'borderRadius': '0 8px 8px 0',
                'position': 'absolute',
                'left': '0',
                'top': '24px',
                'boxShadow': f'0 0 20px {status_color}40'
            }
        ),

        # Contenu
        html.Div([
            # Titre
            html.P(
                title,
                style={
                    'color': COLORS['text_muted'],
                    'fontSize': special_title_style.get('fontSize', '13px'),
                    'fontWeight': '600',
                    'marginBottom': '12px',
                    'textTransform': 'uppercase',
                    'letterSpacing': '0.8px'
                }
            ),

            # Valeur principale
            html.Div([
                html.Span(
                    value,
                    id=value_id,
                    style={
                        'fontSize': special_value_style.get('fontSize', '36px'),
                        'fontWeight': '800',
                        'color': COLORS['text_dark'],
                        'lineHeight': '1',
                        'background': f'linear-gradient(135deg, {COLORS["text_dark"]} 0%, {status_color} 100%)',
                        'WebkitBackgroundClip': 'text',
                        'WebkitTextFillColor': 'transparent',
                        'whiteSpace': special_value_style.get('whiteSpace', 'nowrap'),
                        'wordBreak': special_value_style.get('wordBreak', 'normal')
                    }
                ),
                html.Span(
                    unit,
                    style={
                        'fontSize': '16px',
                        'color': COLORS['text_muted'],
                        'marginLeft': '6px',
                        'fontWeight': '600'
                    }
                ) if unit else None
            ], style={'marginBottom': '16px'}),

            # Tendance avec badge
            html.Div([
                html.Span(
                    [
                        html.Span(trend_icon, style={'marginRight': '4px'}),
                        html.Span("", id=trend_id) if trend_id else html.Span(f"{trend:+.1f}%")
                    ],
                    style={
                        'backgroundColor': f'{trend_color}15',
                        'color': trend_color,
                        'fontSize': '12px',
                        'fontWeight': '700',
                        'padding': '4px 10px',
                        'borderRadius': '6px',
                        'display': 'inline-block'
                    }
                ),
                html.Span(
                    f" {trend_label}",
                    style={
                        'color': COLORS['text_light'],
                        'fontSize': '12px',
                        'marginLeft': '8px'
                    }
                )
            ], style={'marginTop': '8px'})
        ], style={'paddingLeft': '20px', 'position': 'relative', 'zIndex': '1'})
    ], style={
        **CARD_STYLE,
        'position': 'relative',
        'overflow': 'hidden',
        'background': gradient_bg,
        'borderLeft': f'4px solid {status_color}',
        'cursor': 'default',
        'minHeight': extra_min_height if extra_min_height else '180px'
    })

    # add sparkline if provided
    sparkline_elem = None
    if sparkline is not None:
        try:
            fig_sp = create_sparkline(sparkline, color=status_color)
            sparkline_elem = dcc.Graph(
                figure=fig_sp,
                config={'displayModeBar': False},
                style={
                    'height': '50px',
                    'marginTop': '12px',
                    'marginLeft': '16px'
                }
            )
        except Exception:
            pass

    return dbc.Col([card_content, sparkline_elem] if sparkline_elem else [card_content], md=md, lg=lg, xl=xl, className='mb-4')


def create_ratio_chart(data):
    """Graphique évolution ratio combiné modernisé avec gradients"""

    fig = go.Figure()
    x = list(data['mois'])
    y = list(data['ratio'])

    # Moyenne mobile
    try:
        import pandas as _pd
        ma = _pd.Series(y).rolling(window=3, min_periods=1).mean().tolist()
    except Exception:
        ma = y

    # Zone de remplissage avec gradient
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.08)',
        line=dict(color=COLORS['primary'], width=4, shape='spline'),
        mode='lines',
        name='Ratio combiné',
        hovertemplate='<b>%{x}</b><br>Ratio: %{y:.1f}%<extra></extra>'
    ))

    # Ligne principale avec marqueurs
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=10,
            color=y,
            colorscale=[[0, COLORS['success']], [0.5, COLORS['warning']], [1, COLORS['danger']]],
            cmin=95,
            cmax=110,
            line=dict(width=2, color='white')
        ),
        name='Points',
        hovertemplate='<b>%{x}</b><br>Ratio: %{y:.1f}%<extra></extra>',
        showlegend=False
    ))

    # Ligne moyenne mobile élégante
    fig.add_trace(go.Scatter(
        x=x,
        y=ma,
        mode='lines',
        line=dict(color=COLORS['primary_dark'], width=2, dash='dot'),
        name='Tendance (MA3)',
        hovertemplate='Tendance: %{y:.1f}%<extra></extra>',
        opacity=0.7
    ))

    # Dernier point highlight
    if len(x) > 0:
        fig.add_trace(go.Scatter(
            x=[x[-1]],
            y=[y[-1]],
            mode='markers+text',
            marker=dict(size=18, color=COLORS['primary'], line=dict(width=3, color='white')),
            text=[f"<b>{y[-1]:.1f}%</b>"],
            textposition='top center',
            textfont=dict(size=14, color=COLORS['text_dark'], family='Inter'),
            hoverinfo='skip',
            showlegend=False
        ))

    # Zone de danger (au-dessus de 105%)
    fig.add_hrect(
        y0=105, y1=120,
        fillcolor=COLORS['danger_light'],
        opacity=0.3,
        line_width=0,
        annotation_text="Zone Critique",
        annotation_position="top right",
        annotation_font_size=11,
        annotation_font_color=COLORS['danger']
    )

    # Zone d'alerte (100-105%)
    fig.add_hrect(
        y0=100, y1=105,
        fillcolor=COLORS['warning_light'],
        opacity=0.2,
        line_width=0,
        annotation_text="Zone Alerte",
        annotation_position="top right",
        annotation_font_size=11,
        annotation_font_color=COLORS['warning']
    )

    # Ligne de seuil critique
    fig.add_hline(
        y=105,
        line_dash="dash",
        line_color=COLORS['danger'],
        line_width=2.5,
        annotation_text="Seuil Critique (105%)",
        annotation_position="right",
        annotation_font=dict(size=12, color=COLORS['danger'], family='Inter')
    )

    # Ligne équilibre
    fig.add_hline(
        y=100,
        line_dash="dot",
        line_color=COLORS['success'],
        line_width=2,
        annotation_text="Équilibre",
        annotation_position="left",
        annotation_font=dict(size=11, color=COLORS['success'], family='Inter')
    )

    fig.update_layout(
        plot_bgcolor='rgba(248, 250, 252, 0.5)',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40),
        height=350,
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor=COLORS['border'],
            tickfont=dict(color=COLORS['text_muted'], size=11, family='Inter'),
            title=dict(text='Mois', font=dict(size=12, color=COLORS['text_muted']))
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS['border_light'],
            gridwidth=1,
            showline=False,
            tickfont=dict(color=COLORS['text_muted'], size=11, family='Inter'),
            ticksuffix='%',
            title=dict(text='Ratio Combiné (%)', font=dict(size=12, color=COLORS['text_muted']))
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family='Inter',
            bordercolor=COLORS['border']
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=11, color=COLORS['text_muted'], family='Inter'),
            bgcolor='rgba(255,255,255,0.9)'
        )
    )

    return fig


def create_finances_chart(data):
    """Graphique cotisations vs remboursements modernisé"""

    fig = go.Figure()
    x = list(data['mois'])
    cot = list(data['cotisations'])
    remb = list(data['remboursements'])

    # Barres cotisations avec gradient
    fig.add_trace(go.Bar(
        x=x,
        y=cot,
        name='Cotisations',
        marker=dict(
            color=cot,
            colorscale=[[0, COLORS['success_light']], [1, COLORS['success']]],
            line=dict(width=0)
        ),
        hovertemplate='<b>%{x}</b><br>Cotisations: <b>%{y:.1f} M FCFA</b><extra></extra>',
        opacity=0.9
    ))

    # Barres remboursements avec gradient
    fig.add_trace(go.Bar(
        x=x,
        y=remb,
        name='Remboursements',
        marker=dict(
            color=remb,
            colorscale=[[0, COLORS['warning_light']], [1, COLORS['warning']]],
            line=dict(width=0)
        ),
        hovertemplate='<b>%{x}</b><br>Remboursements: <b>%{y:.1f} M FCFA</b><extra></extra>',
        opacity=0.9
    ))

    # Ligne d'écart (cotisations - remboursements)
    diff = [c - r for c, r in zip(cot, remb)]
    try:
        import pandas as _pd
        ma_diff = _pd.Series(diff).rolling(window=3, min_periods=1).mean().tolist()
    except Exception:
        ma_diff = diff

    # Ligne écart avec marqueurs
    fig.add_trace(go.Scatter(
        x=x,
        y=ma_diff,
        mode='lines+markers',
        line=dict(color=COLORS['primary'], width=3, shape='spline'),
        marker=dict(
            size=8,
            color=COLORS['primary'],
            line=dict(width=2, color='white')
        ),
        name='Solde (MA3)',
        yaxis='y2',
        hovertemplate='<b>%{x}</b><br>Solde: <b>%{y:.1f} M FCFA</b><extra></extra>'
    ))

    # Ligne zéro pour le second axe
    fig.add_hline(
        y=0,
        yref='y2',
        line_dash="dot",
        line_color=COLORS['text_light'],
        line_width=1
    )

    fig.update_layout(
        plot_bgcolor='rgba(248, 250, 252, 0.5)',
        paper_bgcolor='white',
        margin=dict(l=40, r=60, t=40, b=40),
        height=380,
        barmode='group',
        bargap=0.25,
        bargroupgap=0.08,
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor=COLORS['border'],
            tickfont=dict(color=COLORS['text_muted'], size=11, family='Inter'),
            title=dict(text='Mois', font=dict(size=12, color=COLORS['text_muted']))
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS['border_light'],
            gridwidth=1,
            showline=False,
            tickfont=dict(color=COLORS['text_muted'], size=11, family='Inter'),
            ticksuffix=' M',
            title=dict(text='Montant (M FCFA)', font=dict(size=12, color=COLORS['text_muted']))
        ),
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False,
            tickfont=dict(color=COLORS['primary'], size=11, family='Inter'),
            ticksuffix=' M',
            title=dict(text='Solde (M FCFA)', font=dict(size=12, color=COLORS['primary']))
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=11, color=COLORS['text_muted'], family='Inter'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=COLORS['border'],
            borderwidth=1
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family='Inter',
            bordercolor=COLORS['border']
        )
    )

    return fig


def create_sparkline(values, color=COLORS['primary']):
    """Retourne une figure sparkline pour insérer dans une carte KPI"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(values))),
        y=values,
        mode='lines',
        line=dict(color=color, width=2),
        hoverinfo='none'
    ))
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        height=40,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_fraud_gauge(value):
    """Jauge de détection fraude (value en %)."""

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': '%', 'font': {'size': 40, 'color': COLORS['text_dark']}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': COLORS['border']},
            'bar': {'color': COLORS['primary']},
            'bgcolor': COLORS['bg_light'],
            'borderwidth': 0,
            'steps': [
                {'range': [0, 60], 'color': '#fef2f2'},
                {'range': [60, 80], 'color': '#fef9c3'},
                {'range': [80, 100], 'color': '#dcfce7'}
            ],
            'threshold': {
                'line': {'color': COLORS['success'], 'width': 3},
                'thickness': 0.8,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='white',
        margin=dict(l=30, r=30, t=30, b=10),
        height=200
    )
    
    return fig


def create_service_distribution_donut(df):
    """Graphique en donut moderne montrant la distribution des types de service"""
    if df.empty:
        return go.Figure()

    service_counts = df.groupby('service_type')['amount'].sum().reset_index()
    service_counts = service_counts.sort_values('amount', ascending=False)

    # Palette de couleurs moderne
    colors_palette = [
        COLORS['primary'], COLORS['success'], COLORS['warning'],
        COLORS['danger'], COLORS['info'], COLORS['purple']
    ]

    fig = go.Figure(data=[go.Pie(
        labels=service_counts['service_type'],
        values=service_counts['amount'],
        hole=0.5,
        marker=dict(
            colors=colors_palette[:len(service_counts)],
            line=dict(color='white', width=3)
        ),
        textinfo='label+percent',
        textfont=dict(size=13, family='Inter, sans-serif', color='white'),
        hovertemplate='<b>%{label}</b><br>Montant: %{value:,.0f} FCFA<br>Part: %{percent}<extra></extra>',
        rotation=45
    )])

    fig.update_layout(
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
        font=dict(family='Inter, sans-serif', size=12, color=COLORS['text_dark']),
        annotations=[dict(
            text=f'<b>{len(service_counts)}</b><br>Services',
            x=0.5, y=0.5,
            font=dict(size=20, color=COLORS['text_dark'], family='Inter, sans-serif'),
            showarrow=False
        )]
    )

    return fig


def create_claims_funnel(df):
    """Graphique en entonnoir moderne pour le workflow des réclamations"""
    if df.empty:
        return go.Figure()

    # Simuler les étapes du funnel basées sur les données
    total_claims = len(df)
    approved = len(df[df['status'] == 'approved']) if 'status' in df.columns else int(total_claims * 0.85)
    in_review = int(total_claims * 0.12)
    rejected = total_claims - approved - in_review

    stages = ['Soumis', 'En révision', 'Approuvés', 'Rejetés']
    values = [total_claims, in_review + approved, approved, rejected]

    colors_funnel = [COLORS['info'], COLORS['warning'], COLORS['success'], COLORS['danger']]

    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(
            color=colors_funnel,
            line=dict(width=2, color='white')
        ),
        connector=dict(line=dict(color=COLORS['border'], width=3)),
        hovertemplate='<b>%{label}</b><br>Dossiers: %{value}<br>%{percentInitial}<extra></extra>'
    ))

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
        font=dict(family='Inter, sans-serif', size=13, color=COLORS['text_dark'])
    )

    return fig


def create_weekly_heatmap(df):
    """Heatmap moderne montrant l'activité des réclamations par jour de la semaine et heure"""
    if df.empty:
        return go.Figure()

    # Simuler des heures (car pas dans les données)
    df_temp = df.copy()
    df_temp['day_of_week'] = df_temp['date'].dt.day_name()
    df_temp['week_num'] = df_temp['date'].dt.isocalendar().week

    # Agrégation par jour de la semaine et semaine
    last_4_weeks = df_temp['week_num'].unique()[-4:] if len(df_temp['week_num'].unique()) >= 4 else df_temp['week_num'].unique()
    df_recent = df_temp[df_temp['week_num'].isin(last_4_weeks)]

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = df_recent.groupby(['week_num', 'day_of_week']).size().reset_index(name='count')

    # Créer matrice pivot
    pivot = heatmap_data.pivot(index='day_of_week', columns='week_num', values='count').fillna(0)
    pivot = pivot.reindex(day_order)

    # Traduire les jours en français
    day_translation = {
        'Monday': 'Lun', 'Tuesday': 'Mar', 'Wednesday': 'Mer',
        'Thursday': 'Jeu', 'Friday': 'Ven', 'Saturday': 'Sam', 'Sunday': 'Dim'
    }
    pivot.index = [day_translation.get(d, d) for d in pivot.index]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f'S{w}' for w in pivot.columns],
        y=pivot.index,
        colorscale=[
            [0, COLORS['bg_light']],
            [0.3, COLORS['primary_light']],
            [0.6, COLORS['primary']],
            [1, COLORS['primary_dark']]
        ],
        text=pivot.values,
        texttemplate='%{text:.0f}',
        textfont=dict(color='white', size=12),
        hovertemplate='<b>%{y}</b> - %{x}<br>Réclamations: %{z}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title=dict(text='Dossiers', side='right'),
            tickmode='linear',
            tick0=0,
            dtick=5
        )
    ))

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=60, r=20, t=20, b=40),
        height=280,
        font=dict(family='Inter, sans-serif', size=12, color=COLORS['text_dark']),
        xaxis=dict(title='Semaine', side='bottom'),
        yaxis=dict(title='')
    )

    return fig


def create_alert_row(alerte):
    """Ligne d'alerte modernisée avec bordure colorée"""

    return html.Div([
        html.Div([
            # Indicateur visuel coloré
            html.Div(
                style={
                    'width': '5px',
                    'height': '100%',
                    'backgroundColor': alerte['color'],
                    'borderRadius': '4px 0 0 4px',
                    'position': 'absolute',
                    'left': '0',
                    'top': '0',
                    'boxShadow': f'0 0 10px {alerte["color"]}50'
                }
            ),
            # Badge priorité
            html.Div(
                alerte['priorite'][0],
                style={
                    'width': '32px',
                    'height': '32px',
                    'borderRadius': '8px',
                    'backgroundColor': f"{alerte['color']}20",
                    'color': alerte['color'],
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'fontWeight': '700',
                    'fontSize': '14px',
                    'marginRight': '14px',
                    'flexShrink': '0'
                }
            ),
            # Contenu
            html.Div([
                html.P(
                    alerte['titre'],
                    style={
                        'margin': '0 0 6px 0',
                        'fontSize': '14px',
                        'color': COLORS['text_dark'],
                        'fontWeight': '600',
                        'lineHeight': '1.4'
                    }
                ),
                html.Div([
                    html.Span(
                        alerte['priorite'],
                        style={
                            'fontSize': '11px',
                            'fontWeight': '700',
                            'color': alerte['color'],
                            'backgroundColor': f"{alerte['color']}15",
                            'padding': '3px 10px',
                            'borderRadius': '6px',
                            'display': 'inline-block',
                            'marginRight': '8px',
                            'textTransform': 'uppercase',
                            'letterSpacing': '0.5px'
                        }
                    ),
                    html.Span(
                        alerte['date'],
                        style={
                            'fontSize': '12px',
                            'color': COLORS['text_light'],
                            'fontWeight': '500'
                        }
                    )
                ])
            ], style={'flex': '1'})
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'padding': '16px 20px',
            'paddingLeft': '24px',
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'marginBottom': '12px',
            'border': f'1px solid {COLORS["border"]}',
            'position': 'relative',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.04)',
            'transition': 'all 0.2s ease',
            'cursor': 'pointer'
        })
    ])


def detect_fraud(df_claims, contamination=0.05, random_state=42):
    """Détecte les dossiers suspects avec IsolationForest sur le montant et délai.
    Ajoute une colonne 'fraud_flag' booléenne et renvoie le DataFrame.
    """
    if df_claims.empty:
        df_claims['fraud_flag'] = False
        return df_claims

    X = df_claims[['amount', 'processing_days']].copy()
    # scale roughly
    X['amount'] = np.log1p(X['amount'])
    X['processing_days'] = X['processing_days']

    iso = IsolationForest(contamination=contamination, random_state=random_state)
    preds = iso.fit_predict(X)
    df_claims = df_claims.copy()
    df_claims['fraud_flag'] = preds == -1
    return df_claims


def compute_kpis_from_simulation(ratio_data, finances_data, claims_df):
    """Calcule approximativement les 10 premiers KPI à partir des données simulées."""
    k = {}

    # Financiers
    total_contrib = sum(finances_data['cotisations'])
    total_claims = sum(finances_data['remboursements'])
    combined_ratio = (total_claims / total_contrib * 100) if total_contrib > 0 else 0
    loss_ratio = (total_claims / total_contrib * 100) if total_contrib > 0 else 0
    cash_balance = 200.0  # M FCFA simulated placeholder
    cash_burn_weeks = 20  # placeholder

    k['total_contributions'] = total_contrib
    k['total_claims_paid'] = total_claims
    k['combined_ratio'] = round(combined_ratio, 1)
    k['loss_ratio'] = round(loss_ratio, 1)
    k['cash_balance'] = cash_balance
    k['cash_burn_rate'] = cash_burn_weeks

    # Adhésions (simulated)
    new_members = int(np.random.poisson(25))
    cancelled_members = int(np.random.poisson(5))
    net_growth = new_members - cancelled_members
    total_active_members = 56284

    k['new_members'] = new_members
    k['cancelled_members'] = cancelled_members
    k['net_growth'] = net_growth
    k['total_active_members'] = total_active_members

    # Prestataires
    active_providers = len(set(claims_df['provider'].unique()))
    active_providers_pharmacies = int(active_providers * 0.6)
    inactive_providers = max(0, 274 - active_providers)

    k['active_providers'] = active_providers
    k['active_providers_pharmacies'] = active_providers_pharmacies
    k['inactive_providers'] = inactive_providers

    # Volume dossiers
    new_claims_submitted = len(claims_df)
    claims_processed = int(new_claims_submitted * 0.9)
    claims_approved = int(claims_processed * 0.88)
    claims_rejected = claims_processed - claims_approved
    claims_paid = int(claims_approved * 0.95)
    pending_claims_count = max(0, new_claims_submitted - claims_processed)

    k['new_claims_submitted'] = new_claims_submitted
    k['claims_processed'] = claims_processed
    k['claims_approved'] = claims_approved
    k['claims_rejected'] = claims_rejected
    k['claims_paid'] = claims_paid
    k['pending_claims_count'] = pending_claims_count

    # Délais
    avg_processing_days = int(claims_df['processing_days'].mean()) if not claims_df.empty else 0
    median_processing_days = int(claims_df['processing_days'].median()) if not claims_df.empty else 0
    pct_over_90 = float((claims_df['processing_days'] > 90).mean() * 100) if not claims_df.empty else 0
    pct_over_180 = float((claims_df['processing_days'] > 180).mean() * 100) if not claims_df.empty else 0

    k['avg_processing_days'] = avg_processing_days
    k['median_processing_days'] = median_processing_days
    k['pct_claims_over_90days'] = round(pct_over_90, 1)
    k['pct_claims_over_180days'] = round(pct_over_180, 1)

    return k


# Initialisation du cache des KPI après définition de la fonction
cached_kpis = compute_kpis_from_simulation(ratio_data, finances_data, claims_df)

# séries mensuelles pour sparklines (utilisées dans le layout)
monthly_claims_mean = claims_df.set_index('date').resample('M')['amount'].mean().fillna(0).round().tolist()
monthly_claims_count = claims_df.set_index('date').resample('M')['amount'].count().tolist()
monthly_processing_days_avg = claims_df.set_index('date').resample('M')['processing_days'].mean().fillna(0).round().tolist()

# synthetic members/providers sparklines
members_sparkline = (np.array([cached_kpis.get('total_active_members', 56284)]) + np.linspace(-200, 200, len(ratio_data['mois']))).astype(int).tolist()
providers_sparkline = (np.array(monthly_claims_count) * 0.01 + 1).tolist()


# ============================================================
# LAYOUT PRINCIPAL
# ============================================================

app.layout = html.Div([
    
    # Header moderne avec gradient et ombre
    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        # Logo MSAE
                        html.Div([
                            html.Div(
                                "MSAE",
                                style={
                                    'fontSize': '28px',
                                    'fontWeight': '900',
                                    'color': 'white',
                                    'letterSpacing': '2px'
                                }
                            )
                        ], style={
                            'width': '72px',
                            'height': '72px',
                            'background': 'linear-gradient(135deg, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0.1) 100%)',
                            'borderRadius': '18px',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'marginRight': '20px',
                            'boxShadow': '0 8px 16px rgba(0,0,0,0.2)',
                            'border': '2px solid rgba(255,255,255,0.3)'
                        }),
                        # Titres
                        html.Div([
                            html.H1(
                                "Tableau de Bord Intelligent",
                                style={
                                    'margin': '0 0 6px 0',
                                    'fontSize': '28px',
                                    'fontWeight': '800',
                                    'color': 'white',
                                    'letterSpacing': '-0.5px'
                                }
                            ),
                            html.P(
                                "Mutuelle de Santé des Agents de l'État du Sénégal",
                                style={
                                    'margin': '0',
                                    'fontSize': '15px',
                                    'color': 'rgba(255,255,255,0.9)',
                                    'fontWeight': '500'
                                }
                            )
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], md=8),
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.Div(
                                "●",
                                style={
                                    'fontSize': '24px',
                                    'color': COLORS['success'],
                                    'animation': 'pulse 2s infinite',
                                    'marginRight': '12px'
                                }
                            ),
                            html.Div([
                                html.P(
                                    "Dernière mise à jour",
                                    style={
                                        'margin': '0',
                                        'fontSize': '11px',
                                        'color': 'rgba(255,255,255,0.8)',
                                        'textTransform': 'uppercase',
                                        'letterSpacing': '0.5px'
                                    }
                                ),
                                html.P(
                                    datetime.now().strftime("%d %B %Y - %H:%M"),
                                    style={
                                        'margin': '0',
                                        'fontSize': '15px',
                                        'fontWeight': '700',
                                        'color': 'white'
                                    }
                                )
                            ])
                        ], style={'display': 'flex', 'alignItems': 'center'})
                    ], style={
                        'textAlign': 'right',
                        'background': 'linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%)',
                        'padding': '16px 24px',
                        'borderRadius': '12px',
                        'border': '1px solid rgba(255,255,255,0.2)',
                        'backdropFilter': 'blur(10px)'
                    })
                    ,
                ], md=4, className='d-flex justify-content-end align-items-center')
            ])
        ], fluid=True)
    ], style={
        'background': f'linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_dark"]} 50%, {COLORS["purple"]} 100%)',
        'padding': '32px 0',
        'marginBottom': '40px',
        'boxShadow': '0 10px 40px rgba(0,0,0,0.15)',
        'position': 'relative',
        'overflow': 'hidden'
    }),
    
    # Contenu principal
    dbc.Container([
        
        # Filtres
        dbc.Row([
            dbc.Col([
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=(datetime.now() - pd.Timedelta(days=365)).date(),
                    end_date=datetime.now().date(),
                    display_format='DD/MM/YYYY'
                )
            ], md=6),
            dbc.Col([
                dcc.Dropdown(
                    id='service-type',
                    options=[{'label': s, 'value': s} for s in sorted(claims_df['service_type'].unique())],
                    value=None,
                    placeholder='Filtrer par type de service',
                    clearable=True
                )
            ], md=6)
        ], className='mb-4'),

        # KPIs (updatables) — 10 premiers indicateurs
        dbc.Row([
            create_kpi_card(
                title="Ratio Combiné",
                value="--",
                unit="%",
                trend=0.0,
                trend_label="vs semaine précédente",
                status_color=COLORS['warning'],
                    id_prefix='kpi-ratio',
                    sparkline=ratio_data['ratio']
            ),
            create_kpi_card(
                title="Adhérents Actifs",
                value="--",
                unit="",
                trend=0.0,
                trend_label="vs semaine précédente",
                status_color=COLORS['success'],
                    id_prefix='kpi-members',
                    sparkline=members_sparkline
            ),
            create_kpi_card(
                title="Prestataires Actifs",
                value="--",
                unit="",
                trend=0.0,
                trend_label="vs semaine précédente",
                status_color=COLORS['danger'],
                    id_prefix='kpi-providers',
                    sparkline=providers_sparkline
            ),
            create_kpi_card(
                title="Délai Moyen",
                value="--",
                unit="jours",
                trend=0.0,
                trend_label="vs semaine précédente",
                status_color=COLORS['info'],
                    id_prefix='kpi-delay',
                    sparkline=monthly_processing_days_avg
            ),
        ]),

        dbc.Row([
            create_kpi_card(
                title="Cotisations Totales",
                value="--",
                unit="M FCFA",
                trend=0.0,
                trend_label="vs semaine précédente",
                status_color=COLORS['primary'],
                id_prefix='kpi-cotisations',
                sparkline=finances_data['cotisations']
            ),
            create_kpi_card(
                title="Remboursements Totaux",
                value="--",
                unit="M FCFA",
                trend=0.0,
                trend_label="vs semaine précédente",
                status_color=COLORS['primary_dark'],
                id_prefix='kpi-remboursements',
                sparkline=finances_data['remboursements']
            ),
            create_kpi_card(
                title="Coût Moyen par Dossier",
                value="--",
                unit="FCFA",
                trend=0.0,
                trend_label="vs précédent",
                status_color=COLORS['warning'],
                id_prefix='kpi-cost-claim',
                sparkline=monthly_claims_mean
            ),
            create_kpi_card(
                title="Nombre Dossiers",
                value="--",
                unit="",
                trend=0.0,
                trend_label="vs semaine précédente",
                status_color=COLORS['success'],
                id_prefix='kpi-claims-count',
                sparkline=monthly_claims_count
            ),
        ]),

        dbc.Row([
            create_kpi_card(
                title="Taux Fraude",
                value="--",
                unit="%",
                trend=0.0,
                trend_label="vs semaine précédente",
                status_color=COLORS['danger'],
                    id_prefix='kpi-fraud-rate',
                    sparkline=ratio_data['ratio'],
                    md=12, lg=12, xl=12
            ),
            create_kpi_card(
                title="Top Service (part)",
                value="--",
                unit="",
                trend=0.0,
                trend_label="part du total",
                status_color=COLORS['info'],
                    id_prefix='kpi-top-service',
                    sparkline=monthly_claims_count,
                    md=12, lg=12, xl=12
            ),
        ]),
        
        # Graphiques
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3(
                        "Évolution du Ratio Combiné",
                        style={
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'color': COLORS['text_dark'],
                            'marginBottom': '4px'
                        }
                    ),
                    html.P(
                        "12 derniers mois",
                        style={
                            'fontSize': '13px',
                            'color': COLORS['text_muted'],
                            'marginBottom': '20px'
                        }
                    ),
                    dcc.Graph(
                        id='graph-ratio',
                        figure=create_ratio_chart(ratio_data),
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '400px'}
                    )
                ], style=CARD_STYLE)
            ], md=6, className='mb-4'),
            
            dbc.Col([
                html.Div([
                    html.H3(
                        "Cotisations vs Remboursements",
                        style={
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'color': COLORS['text_dark'],
                            'marginBottom': '4px'
                        }
                    ),
                    html.P(
                        "6 derniers mois (en millions FCFA)",
                        style={
                            'fontSize': '13px',
                            'color': COLORS['text_muted'],
                            'marginBottom': '20px'
                        }
                    ),
                    dcc.Graph(
                        id='graph-finances',
                        figure=create_finances_chart(finances_data),
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '400px'}
                    )
                ], style=CARD_STYLE)
            ], md=6, className='mb-4'),
        ]),

        # Nouveaux graphiques modernes - Distribution et Analyse
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3(
                        "Distribution des Services",
                        style={
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'color': COLORS['text_dark'],
                            'marginBottom': '4px'
                        }
                    ),
                    html.P(
                        "Répartition par type de service",
                        style={
                            'fontSize': '13px',
                            'color': COLORS['text_muted'],
                            'marginBottom': '20px'
                        }
                    ),
                    dcc.Graph(
                        id='graph-service-donut',
                        figure=create_service_distribution_donut(claims_df),
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '350px'}
                    )
                ], style=CARD_STYLE)
            ], md=4, className='mb-4'),

            dbc.Col([
                html.Div([
                    html.H3(
                        "Workflow des Réclamations",
                        style={
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'color': COLORS['text_dark'],
                            'marginBottom': '4px'
                        }
                    ),
                    html.P(
                        "Progression des dossiers",
                        style={
                            'fontSize': '13px',
                            'color': COLORS['text_muted'],
                            'marginBottom': '20px'
                        }
                    ),
                    dcc.Graph(
                        id='graph-claims-funnel',
                        figure=create_claims_funnel(claims_df),
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '350px'}
                    )
                ], style=CARD_STYLE)
            ], md=4, className='mb-4'),

            dbc.Col([
                html.Div([
                    html.H3(
                        "Activité Hebdomadaire",
                        style={
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'color': COLORS['text_dark'],
                            'marginBottom': '4px'
                        }
                    ),
                    html.P(
                        "Heatmap des 4 dernières semaines",
                        style={
                            'fontSize': '13px',
                            'color': COLORS['text_muted'],
                            'marginBottom': '20px'
                        }
                    ),
                    dcc.Graph(
                        id='graph-weekly-heatmap',
                        figure=create_weekly_heatmap(claims_df),
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '350px'}
                    )
                ], style=CARD_STYLE)
            ], md=4, className='mb-4'),
        ]),

        # Alertes et Fraude
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3(
                                "Alertes Actives",
                                style={
                                    'fontSize': '16px',
                                    'fontWeight': '600',
                                    'color': COLORS['text_dark'],
                                    'margin': '0'
                                }
                            ),
                            html.P(
                                f"{len(alertes)} alertes en attente",
                                style={
                                    'fontSize': '13px',
                                    'color': COLORS['text_muted'],
                                    'margin': '0'
                                }
                            )
                        ]),
                        html.A(
                            "Voir tout",
                            href="#",
                            style={
                                'fontSize': '13px',
                                'fontWeight': '600',
                                'color': COLORS['primary'],
                                'textDecoration': 'none'
                            }
                        )
                    ], style={
                        'display': 'flex',
                        'justifyContent': 'space-between',
                        'alignItems': 'center',
                        'marginBottom': '20px'
                    }),
                    html.Div([create_alert_row(a) for a in alertes])
                ], style=CARD_STYLE)
            ], md=6, className='mb-4'),
            
            dbc.Col([
                html.Div([
                    html.H3(
                        "Détection Fraude",
                        style={
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'color': COLORS['text_dark'],
                            'marginBottom': '4px'
                        }
                    ),
                    html.P(
                        "Précision du modèle ML",
                        style={
                            'fontSize': '13px',
                            'color': COLORS['text_muted'],
                            'marginBottom': '10px'
                        }
                    ),
                    dcc.Graph(
                        id='fraud-gauge',
                        figure=create_fraud_gauge(87),
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '250px'}
                    ),
                    html.Div([
                        html.Div([
                            html.Span(
                                "--",
                                id='fraud-count',
                                style={
                                    'fontSize': '28px',
                                    'fontWeight': '700',
                                    'color': COLORS['warning']
                                }
                            ),
                            html.P(
                                "Dossiers suspects",
                                style={
                                    'fontSize': '12px',
                                    'color': COLORS['text_muted'],
                                    'margin': '0'
                                }
                            )
                        ], style={'textAlign': 'center'}),
                        html.Div([
                            html.Span(
                                "--",
                                id='fraud-rate',
                                style={
                                    'fontSize': '28px',
                                    'fontWeight': '700',
                                    'color': COLORS['primary']
                                }
                            ),
                            html.P(
                                "Taux détection",
                                style={
                                    'fontSize': '12px',
                                    'color': COLORS['text_muted'],
                                    'margin': '0'
                                }
                            )
                        ], style={'textAlign': 'center'})
                    ], style={
                        'display': 'flex',
                        'justifyContent': 'space-around',
                        'paddingTop': '16px',
                        'borderTop': f'1px solid {COLORS["border"]}',
                        'marginTop': '16px'
                    }),

                    # Tableau des dossiers suspects
                    html.Div(style={'marginTop': '16px'}, children=[
                        dash.dash_table.DataTable(
                            id='fraud-table',
                            columns=[
                                {'name': 'date', 'id': 'date'},
                                {'name': 'service_type', 'id': 'service_type'},
                                {'name': 'provider', 'id': 'provider'},
                                {'name': 'amount', 'id': 'amount'},
                                {'name': 'processing_days', 'id': 'processing_days'}
                            ],
                            data=[],
                            page_size=10,
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left'}
                        )
                    ])
                ], style=CARD_STYLE)
            ], md=6, className='mb-4'),
        ]),
        
        # Footer
        html.Footer([
            html.Hr(style={'borderColor': COLORS['border'], 'marginTop': '32px'}),
            html.P(
                "© 2025 MSAE - Tableau de Bord Intelligent | Marc MARE & Awa GUEYE",
                style={
                    'textAlign': 'center',
                    'color': COLORS['text_muted'],
                    'fontSize': '13px',
                    'padding': '16px 0'
                }
            )
        ])
        
    ], fluid=True)
    
], style={
    'backgroundColor': COLORS['bg_light'],
    'minHeight': '100vh',
    'fontFamily': "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
})


# ============================================================
# CALLBACKS
# ============================================================


@app.callback(
    [
        # KPI values & trends (4 existantes + 6 nouvelles = 10 KPI)
        dash.Output('kpi-ratio-value', 'children'),
        dash.Output('kpi-ratio-trend', 'children'),
        dash.Output('kpi-members-value', 'children'),
        dash.Output('kpi-members-trend', 'children'),
        dash.Output('kpi-providers-value', 'children'),
        dash.Output('kpi-providers-trend', 'children'),
        dash.Output('kpi-delay-value', 'children'),
        dash.Output('kpi-delay-trend', 'children'),

        dash.Output('kpi-cotisations-value', 'children'),
        dash.Output('kpi-cotisations-trend', 'children'),
        dash.Output('kpi-remboursements-value', 'children'),
        dash.Output('kpi-remboursements-trend', 'children'),
        dash.Output('kpi-cost-claim-value', 'children'),
        dash.Output('kpi-cost-claim-trend', 'children'),
        dash.Output('kpi-claims-count-value', 'children'),
        dash.Output('kpi-claims-count-trend', 'children'),
        dash.Output('kpi-fraud-rate-value', 'children'),
        dash.Output('kpi-fraud-rate-trend', 'children'),
        dash.Output('kpi-top-service-value', 'children'),
        dash.Output('kpi-top-service-trend', 'children'),

        # Figures
        dash.Output('graph-ratio', 'figure'),
        dash.Output('graph-finances', 'figure'),

        # New modern graphs
        dash.Output('graph-service-donut', 'figure'),
        dash.Output('graph-claims-funnel', 'figure'),
        dash.Output('graph-weekly-heatmap', 'figure'),

        # Fraud panel
        dash.Output('fraud-gauge', 'figure'),
        dash.Output('fraud-count', 'children'),
        dash.Output('fraud-rate', 'children'),
        dash.Output('fraud-table', 'data')
    ],
    [
        dash.Input('date-range', 'start_date'),
        dash.Input('date-range', 'end_date'),
        dash.Input('service-type', 'value')
    ]
)
def update_dashboard(start_date, end_date, service_type):
    # filter claims
    df = claims_df.copy()
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date) + pd.Timedelta(days=1)]
    if service_type:
        df = df[df['service_type'] == service_type]

    # Use compute_kpis_from_simulation to get rich simulated KPIs (on filtered claims)
    k = compute_kpis_from_simulation(ratio_data, finances_data, df)

    # Basic KPIs (existing)
    def pct_change(curr, prev):
        try:
            if prev == 0:
                return 0.0
            return (curr - prev) / abs(prev) * 100.0
        except Exception:
            return 0.0

    # Ratio: use last month values from ratio_data for month-over-month
    try:
        ratio_curr = float(ratio_data['ratio'][-1])
        ratio_prev = float(ratio_data['ratio'][-2]) if len(ratio_data['ratio']) >= 2 else ratio_curr
    except Exception:
        ratio_curr = k.get('combined_ratio', 0)
        ratio_prev = ratio_curr
    ratio_val = ratio_curr
    kpi_ratio_trend_val = pct_change(ratio_curr, ratio_prev)

    # Members
    members = k.get('total_active_members', 0)
    members_prev = max(0, members - k.get('net_growth', 0))
    members_trend_pct = pct_change(members, members_prev)

    # Providers: compute week-over-week unique providers in filtered df
    try:
        if not df.empty:
            weeks = df['date'].dt.to_period('W')
            last_w = weeks.max()
            prev_w = last_w - 1
            providers_curr = int(df[weeks == last_w]['provider'].nunique())
            providers_prev = int(df[weeks == prev_w]['provider'].nunique()) if prev_w in weeks.values else providers_curr
        else:
            providers_curr = k.get('active_providers', 0)
            providers_prev = providers_curr
    except Exception:
        providers_curr = k.get('active_providers', 0)
        providers_prev = providers_curr
    providers = providers_curr
    providers_trend = pct_change(providers_curr, providers_prev)

    # Delay (avg processing days) week-over-week
    try:
        if not df.empty:
            avg_delay_curr = int(df['processing_days'].mean())
            weeks = df['date'].dt.to_period('W')
            last_w = weeks.max()
            prev_w = last_w - 1
            avg_delay_prev = int(df[weeks == prev_w]['processing_days'].mean()) if prev_w in weeks.values else avg_delay_curr
        else:
            avg_delay_curr = k.get('avg_processing_days', 0)
            avg_delay_prev = avg_delay_curr
    except Exception:
        avg_delay_curr = k.get('avg_processing_days', 0)
        avg_delay_prev = avg_delay_curr
    avg_delay = avg_delay_curr
    avg_delay_trend = pct_change(avg_delay_curr, avg_delay_prev)

    # fraud detection
    df_flagged = detect_fraud(df, contamination=0.05)
    fraud_count = int(df_flagged['fraud_flag'].sum())
    fraud_rate_pct = (df_flagged['fraud_flag'].mean() * 100) if not df_flagged.empty else 0.0
    fraud_rate = f"{fraud_rate_pct:.1f}%"

    # Recompute mini time-series for charts from filtered claims (monthly aggregation)
    # For ratio chart we keep monthly ratio_data but could be adapted; use global ratio_data for simplicity
    ratio_fig = create_ratio_chart(ratio_data)
    finances_fig = create_finances_chart(finances_data)

    # New modern graphs based on filtered data
    donut_fig = create_service_distribution_donut(df)
    funnel_fig = create_claims_funnel(df)
    heatmap_fig = create_weekly_heatmap(df)

    fraud_fig = create_fraud_gauge(min(100, max(0, int(fraud_flag_pct := (df_flagged['fraud_flag'].mean() * 100) if not df_flagged.empty else 0))))

    # fraud table data: show top flagged by amount
    fraud_table = df_flagged.loc[df_flagged['fraud_flag']].copy()
    if not fraud_table.empty:
        fraud_table = fraud_table.sort_values('amount', ascending=False).head(50)
        fraud_table['date'] = fraud_table['date'].dt.strftime('%Y-%m-%d')
        table_data = fraud_table[['date', 'service_type', 'provider', 'amount', 'processing_days']].to_dict('records')
    else:
        table_data = []

    # Additional KPI values (from simulated summary)
    cotisations_total = k.get('total_contributions', 0.0)
    remb_total = k.get('total_claims_paid', 0.0)
    cost_per_claim = int(df['amount'].mean()) if not df.empty else 0
    claims_count = k.get('new_claims_submitted', 0)

    # top service
    if not df.empty:
        svc = df.groupby('service_type')['amount'].sum()
        top_service = svc.idxmax()
        top_service_pct = svc.max() / svc.sum() * 100 if svc.sum() > 0 else 0
    else:
        top_service = '--'
        top_service_pct = 0

    # simple month-on-month trends for finances if possible
    try:
        last_cot = float(finances_data['cotisations'][-1])
        prev_cot = float(finances_data['cotisations'][-2]) if len(finances_data['cotisations']) >= 2 else last_cot
    except Exception:
        last_cot = cotisations_total
        prev_cot = last_cot
    cot_trend = pct_change(last_cot, prev_cot)
    try:
        last_remb = float(finances_data['remboursements'][-1])
        prev_remb = float(finances_data['remboursements'][-2]) if len(finances_data['remboursements']) >= 2 else last_remb
    except Exception:
        last_remb = remb_total
        prev_remb = last_remb
    remb_trend = pct_change(last_remb, prev_remb)

    # claims week-over-week: compute for last and previous week from filtered df
    try:
        if not df.empty:
            weeks = df['date'].dt.to_period('W')
            last_w = weeks.max()
            prev_w = last_w - 1
            claims_curr = int(df[weeks == last_w].shape[0])
            claims_prev = int(df[weeks == prev_w].shape[0]) if prev_w in weeks.values else claims_curr
        else:
            claims_curr = claims_count
            claims_prev = claims_count
    except Exception:
        claims_curr = claims_count
        claims_prev = claims_count
    claims_trend = pct_change(claims_curr, claims_prev)

    # cost per claim week-over-week
    try:
        if not df.empty:
            weeks = df['date'].dt.to_period('W')
            last_w = weeks.max()
            prev_w = last_w - 1
            cost_curr = df[weeks == last_w]['amount'].mean() if (weeks == last_w).any() else cost_per_claim
            cost_prev = df[weeks == prev_w]['amount'].mean() if prev_w in weeks.values else cost_curr
        else:
            cost_curr = cost_per_claim
            cost_prev = cost_per_claim
    except Exception:
        cost_curr = cost_per_claim
        cost_prev = cost_per_claim
    cost_trend = pct_change(cost_curr, cost_prev)

    # fraud rate previous period (weekly)
    try:
        if not df.empty:
            weeks = df['date'].dt.to_period('W')
            last_w = weeks.max()
            prev_w = last_w - 1
            df_prev = df[weeks == prev_w]
            if not df_prev.empty:
                fraud_prev = detect_fraud(df_prev, contamination=0.05)['fraud_flag'].mean() * 100
            else:
                fraud_prev = fraud_rate_pct
        else:
            fraud_prev = fraud_rate_pct
    except Exception:
        fraud_prev = fraud_rate_pct
    fraud_rate_trend = pct_change(fraud_rate_pct, fraud_prev)

    # top service previous share (weekly)
    try:
        if not df.empty:
            weeks = df['date'].dt.to_period('W')
            last_w = weeks.max()
            prev_w = last_w - 1
            svc_prev = df[weeks == prev_w].groupby('service_type')['amount'].sum()
            top_prev_pct = svc_prev.max() / svc_prev.sum() * 100 if not svc_prev.empty and svc_prev.sum() > 0 else 0
        else:
            top_prev_pct = top_service_pct
    except Exception:
        top_prev_pct = top_service_pct

    # prepare KPI texts and trends
    kpi_ratio_value = f"{ratio_val:.1f}%"
    kpi_ratio_trend = f"{kpi_ratio_trend_val:+.1f}%"

    kpi_members_value = f"{members:,}"
    kpi_members_trend = f"{members_trend_pct:+.1f}%"

    kpi_providers_value = f"{providers}" 
    kpi_providers_trend = f"{providers_trend:+.1f}%"

    kpi_delay_value = f"{avg_delay}"
    kpi_delay_trend = f"{avg_delay_trend:+.1f}%"

    kpi_cotisations_value = f"{cotisations_total:.1f}"
    kpi_cotisations_trend = f"{cot_trend:+.1f}%"

    kpi_remb_value = f"{remb_total:.1f}"
    kpi_remb_trend = f"{remb_trend:+.1f}%"

    kpi_cost_claim_value = f"{int(cost_curr):,}"
    kpi_cost_claim_trend = f"{cost_trend:+.1f}%"

    kpi_claims_count_value = f"{claims_curr}"
    kpi_claims_count_trend = f"{claims_trend:+.1f}%"

    kpi_fraud_rate_value = f"{fraud_rate_pct:.1f}%"
    kpi_fraud_rate_trend = f"{fraud_rate_trend:+.1f}%"

    kpi_top_service_value = f"{top_service} ({top_service_pct:.1f}%)"
    kpi_top_service_trend = f"{(top_service_pct - top_prev_pct):+.1f}%"

    return (
        kpi_ratio_value,
        kpi_ratio_trend,
        kpi_members_value,
        kpi_members_trend,
        kpi_providers_value,
        kpi_providers_trend,
        kpi_delay_value,
        kpi_delay_trend,

        kpi_cotisations_value,
        kpi_cotisations_trend,
        kpi_remb_value,
        kpi_remb_trend,
        kpi_cost_claim_value,
        kpi_cost_claim_trend,
        kpi_claims_count_value,
        kpi_claims_count_trend,
        kpi_fraud_rate_value,
        kpi_fraud_rate_trend,
        kpi_top_service_value,
        kpi_top_service_trend,

        ratio_fig,
        finances_fig,
        donut_fig,
        funnel_fig,
        heatmap_fig,
        fraud_fig,
        str(fraud_count),
        fraud_rate,
        table_data
    )

# ============================================================
# LANCEMENT
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  MSAE - Tableau de Bord Intelligent")
    print("  Démarrage sur http://localhost:8050")
    print("="*60 + "\n")
    # Démarrer le scheduler hebdomadaire (si disponible)
    try:
        sched = start_weekly_scheduler()
    except Exception as e:
        print(f"[startup] Erreur lors du démarrage du scheduler : {e}")

    # Pour accélérer le démarrage en développement, désactiver le reloader
    # (le reloader crée un second processus et augmente le temps de démarrage)
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=8050)

