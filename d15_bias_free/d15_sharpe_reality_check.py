"""
D15 Sharpe Reality Check — Sind Sharpes > 2 realistisch?
"""
import pandas as pd
import numpy as np

df = pd.read_parquet('d14_bias_free/results/d14_all_trades.parquet')
meta = pd.read_parquet('data/metadata/metadata_v9.parquet')
meta['date'] = meta['date'].astype(str)

oos = df[df['sample'] == 'OOS'].copy()
STRATS = ['1.2_Pre10am_Momentum','2.3_Dip_and_Rip','1.4_Power_Setup',
          '1.3_Earnings_OD_RVOL','1.8_Gap_3ADR','1.5_Big_First_Candle',
          '1.1_S4_Optimiert','1.9_RVOL5_RSI50','1.6_Low_Wick','1.7_Big_Body']
oos = oos[oos['strategy'].isin(STRATS)]

all_dates = sorted(meta[(meta['date'] >= '2024-01-01') & (meta['date'] <= '2026-02-21')]['date'].unique())
n_trading_days = len(all_dates)
n_years = n_trading_days / 252

print('=' * 80)
print('SHARPE-REALITY-CHECK: Sind Sharpes > 2 utopisch?')
print('=' * 80)
print()
print(f'OOS: {n_trading_days} Handelstage ({n_years:.1f} Jahre)')
print()

# ===================================================================
# VERGLEICH: Branchen-Sharpes
# ===================================================================
print('BRANCHEN-VERGLEICH (annualisierte Sharpe):')
print('  S&P 500 Buy&Hold:           ~0.40')
print('  Durchschnittlicher HF:      ~0.50-0.80')
print('  Guter Systematic HF:        ~1.00-1.50')
print('  Top Quant (Renaissance):    ~2.0-3.0 nach Kosten')
print('  HFT Market Making:          ~5-15 (winzige Kapazitaet)')
print()

# ===================================================================
# UNSERE SHARPES: 3 Berechnungsmethoden
# ===================================================================
print('UNSERE STRATEGIEN — 3 BERECHNUNGSMETHODEN:')
print('-' * 100)
header = (f'{"Strategie":<28} | {"N":>5} | {"TrDays":>6} | {"Aktiv%":>6} | '
          f'{"D15_Sha":>7} | {"Daily0":>7} | {"DailyAct":>8} | {"TradesJ":>7}')
print(header)
print('-' * 100)

for strat in STRATS:
    sub = oos[oos['strategy'] == strat]
    n = len(sub)
    daily = sub.groupby('date')['pnl_r'].sum()

    # Methode 1: D15 per-trade Sharpe (inflationiert)
    arr = sub['pnl_r'].values
    ev = np.mean(arr)
    std = np.std(arr, ddof=1)
    per_trade_sha = ev / std if std > 0 else 0
    trades_yr = n / n_years
    d15_sha = per_trade_sha * np.sqrt(trades_yr)

    # Methode 2: Daily 0-filled Sharpe
    full = pd.Series(0.0, index=all_dates)
    full.update(daily)
    mean_d = full.mean()
    std_d = full.std()
    daily0_sha = mean_d / std_d * np.sqrt(252) if std_d > 0 else 0

    # Methode 3: Daily Sharpe NUR auf aktive Tage
    tr_days = len(daily[daily != 0])
    active_pct = tr_days / n_trading_days
    active_only = daily[daily != 0]
    if len(active_only) > 5:
        mean_a = active_only.mean()
        std_a = active_only.std()
        daily_act_sha = mean_a / std_a * np.sqrt(252) if std_a > 0 else 0
    else:
        daily_act_sha = 0

    print(f'{strat:<28} | {n:>5} | {tr_days:>6} | {active_pct:>5.0%} | '
          f'{d15_sha:>7.2f} | {daily0_sha:>7.2f} | {daily_act_sha:>8.2f} | {trades_yr:>7.0f}')

print()
print('Legende:')
print('  D15_Sha   = per_trade_sharpe * sqrt(trades/yr) — D15 Methode (INFLATIONIERT)')
print('  Daily0    = Daily Sharpe mit 0-Tagen (KONSERVATIVSTE, korrekt)')
print('  DailyAct  = Daily Sharpe NUR auf aktive Tage (INFLATIONIERT, ignoriert tote Tage)')
print('  TradesJ   = Trades pro Jahr')
print()

# ===================================================================
# DAS KERN-PROBLEM: R-Sharpe vs. Konto-Sharpe
# ===================================================================
print()
print('=' * 80)
print('DAS KERN-PROBLEM: R-Sharpe ist NICHT gleich Portfolio-Sharpe')
print('=' * 80)
print()
print('Alle obigen Sharpes sind auf R-MULTIPLES berechnet.')
print('R-Multiples sind NORMIERTE Renditen (1R = Risiko pro Trade).')
print()
print('Portfolio-Sharpe haengt ab vom POSITION SIZING:')
print()

# Portfolio-Level
port_daily = oos.groupby('date')['pnl_r'].sum()
port_full = pd.Series(0.0, index=all_dates)
port_full.update(port_daily)

active_days = (port_full != 0).sum()
port_mean = port_full.mean()
port_std = port_full.std()
port_sharpe_r = port_mean / port_std * np.sqrt(252) if port_std > 0 else 0

print(f'Portfolio (10 Strats, 1R/Trade):')
print(f'  Aktive Tage: {active_days}/{n_trading_days} ({active_days/n_trading_days:.0%})')
print(f'  Mean daily PnL: {port_mean:+.2f}R')
print(f'  Std daily PnL:  {port_std:.2f}R')
print(f'  Sharpe (R-basiert): {port_sharpe_r:.2f}')
print(f'  Max daily loss: {port_full.min():+.1f}R')
print(f'  Max daily gain: {port_full.max():+.1f}R')
print()

print('Umrechnung in Konto-Rendite bei verschiedenem Position Sizing:')
print(f'{"1R =":<12} | {"Jahresrend.":>12} | {"Konto-Sharpe":>12} | {"Max Tagesverlust":>16} | {"3-Sigma Tag":>12}')
print('-' * 80)

for risk_pct in [0.10, 0.25, 0.50, 1.0, 2.0]:
    port_pct = port_full * risk_pct  # daily return in %
    ann_ret = port_pct.mean() * 252
    sha = port_pct.mean() / port_pct.std() * np.sqrt(252) if port_pct.std() > 0 else 0
    max_loss = port_pct.min()
    sigma3 = (port_pct.mean() - 3 * port_pct.std())
    print(f'  {risk_pct:.2f}% Konto | {ann_ret:>+11.1f}% | {sha:>12.2f} | {max_loss:>+15.1f}% | {sigma3:>+11.1f}%')

print()
print('WICHTIG: Die Konto-Sharpe ist IDENTISCH zur R-Sharpe!')
print('(Linearer Skalierungsfaktor kuerzt sich in mean/std raus)')
print('Die Frage ist also: Ist eine R-Sharpe von 6.2 realistisch?')
print()

# ===================================================================
# WARUM HOHE SHARPE BEI NIEDRIGEM KAPITAL-EINSATZ
# ===================================================================
print()
print('=' * 80)
print('ERKLAERUNG: WARUM HOHE SHARPE TROTZDEM NICHT "REICH MACHT"')
print('=' * 80)
print()
print('Die Sharpe-Ratio misst Rendite/Risiko — NICHT absolute Rendite.')
print()
print('Beispiel: 2.3 Dip&Rip')
sub = oos[oos['strategy'] == '2.3_Dip_and_Rip']
daily_23 = sub.groupby('date')['pnl_r'].sum()
full_23 = pd.Series(0.0, index=all_dates)
full_23.update(daily_23)
print(f'  Daily Sharpe: {full_23.mean() / full_23.std() * np.sqrt(252):.2f}')
print(f'  Mean daily PnL: {full_23.mean():+.3f}R')
print(f'  Std daily PnL:  {full_23.std():.3f}R')
print()
print(f'  Bei 1R = 0.25% Konto (konservativ):')
print(f'    Mean daily return: {full_23.mean() * 0.25:+.4f}%')
print(f'    Std daily return:  {full_23.std() * 0.25:.4f}%')
print(f'    Annual return:     {full_23.mean() * 0.25 * 252:+.1f}%')
print(f'    Max daily loss:    {full_23.min() * 0.25:+.2f}%')
print()
print('  => Bei konservativem Sizing: +72% Jahresrendite bei winzigem Risiko')
print('  => Das KLINGT nach hoher Sharpe, IST aber kapazitaetsbeschraenkt:')
print(f'    Bei $100k Konto und 1R=0.25%=$250 Risiko/Trade')
print(f'    Max ~5 gleichzeitige Trades = $1,250 Risiko = 1.25% des Kontos')
print(f'    Jahresgewinn: ~$27,000 (gut, aber kein Reichtum)')
print()

# ===================================================================
# ENTSCHEIDENDE FRAGE: Backtest vs Live
# ===================================================================
print()
print('=' * 80)
print('DIE ENTSCHEIDENDE FRAGE: BACKTEST vs LIVE')
print('=' * 80)
print()
print('Typische Sharpe-Decay Faktoren (Backtest => Live):')
print()
print('  Faktor                        | Auswirkung | Auf unsere Sharpe')
print('  ----------------------------------------------------------------')
print('  Slippage-Unterschaetzung      |  -15-25%   | Base=0.05 ADR koennte 0.07 sein')
print('  Ausfuehrungsprobleme          |  -10-20%   | SL-Slides, Late Fills')
print('  Regime-Wechsel                |  -10-30%   | 2024-25 war Gapper-freundlich')
print('  Psychologie/Fehler            |  -5-15%    | Verpasste Trades, Overrides')
print('  Technische Ausfaelle          |  -5-10%    | API-Fehler, Datenluecken')
print('  ----------------------------------------------------------------')
print('  KUMULIERT:                    |  -40-70%   |')
print()

# Berechne Live-Schaetzung
print('LIVE-SCHAETZUNG (Einzelstrategien, Daily Sharpe):')
print(f'{"Strategie":<28} | {"Backtest":>8} | {"Live 50%":>8} | {"Live 30%":>8}')
print('-' * 60)

for strat in STRATS:
    sub = oos[oos['strategy'] == strat]
    daily = sub.groupby('date')['pnl_r'].sum()
    full = pd.Series(0.0, index=all_dates)
    full.update(daily)
    sha = full.mean() / full.std() * np.sqrt(252) if full.std() > 0 else 0
    print(f'{strat:<28} | {sha:>8.2f} | {sha*0.50:>8.2f} | {sha*0.30:>8.2f}')

# Portfolio
print(f'{"PORTFOLIO (10 Strats)":<28} | {port_sharpe_r:>8.2f} | {port_sharpe_r*0.50:>8.2f} | {port_sharpe_r*0.30:>8.2f}')

print()
print()
print('=' * 80)
print('FAZIT')
print('=' * 80)
print()
print('Sind Sharpes von 2-4 "utopisch" fuer systematisches Trading?')
print()
print('NEIN — fuer kapazitaetsbeschraenkte Intraday-Strategien sind')
print('Backtest-Sharpes von 2-4 ERWARTBAR. Das ist kein Bug, sondern')
print('die mathematische Konsequenz von:')
print('  1. Kurze Haltezeit (6h statt 24h) => weniger Varianz')
print('  2. Selektives Trading (nur an Gapper-Tagen) => hoeherer Edge pro Trade')
print('  3. Kleine Kapazitaet (max 5-10 Trades/Tag) => nicht skalierbar')
print()
print('ABER: Im Live-Handel werden diese Sharpes um 40-70% sinken.')
print('Realistische Live-Erwartung:')
print('  - Einzelstrategie: Sharpe 0.8 - 2.0')
print('  - Portfolio (4-5 Strats): Sharpe 1.5 - 3.0')
print()
print('Zum Vergleich:')
print('  - Sharpe 1.5 nach Kosten = Top 5% aller Systematic Trader')
print('  - Sharpe 2.0 nach Kosten = Top 1% (Renaissance-Niveau)')
print('  - Sharpe 3.0 nach Kosten = Unrealistisch ausser HFT')
print()
print('BOTTOM LINE:')
print('  Die Backtest-Sharpes sind NICHT utopisch fuer dieses Segment.')
print('  Die Live-Sharpes werden DEUTLICH niedriger sein.')
print('  Eine Live-Sharpe von 1.0-2.0 waere ein herausragendes Ergebnis.')
