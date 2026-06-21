# D13 Team Memory - Zentrale Wissensbasis

## Phase 1: OD-Strategien (ABGESCHLOSSEN)
| Hyp | Name | EV_OOS(R) | N_OOS | P_OOS |
|-----|------|-----------|-------|-------|
| H1 | Gap 3.0+ ADR, od>0.5 | +0.54R | 506 | 0.0000 |
| H4 | RVOL>=5+RSI>50+GapUp | +0.64R | 266 | 0.0000 |

## Phase 2: Unabhaengige Strategien

### S4 Optimiert = OOS-BESTAETIGT
**Beste Variante: od>=0.7, gap>=2.0 ADR, pm_rth5<0.30**
- Entry: close_935, SL: 0.25*ADR, TRAIL_D, Richtung=gap_direction
- OOS: N=62, EV=+0.620R, WR=62.9%, PF=2.67, P=0.0003

### Ansatz 1: Late-Day Swing (blind) = KEIN EDGE
- d13_lateday_v2.py: Blinder Close-Entry ohne Pattern -> ALLE Halteperioden negativ
- **FAZIT: Blinder Late-Day Entry ohne Pattern-Confirmation hat keinen Edge**

### Ansatz 5: Late-Day Pattern-Confirmed Swing = OOS-BESTAETIGT (TRAIL_D)
**Script: d13_lateday_v3.py | Ergebnis: d13_lateday_v3_results.txt**
- Entry: Letzte 90 Min (14:30/15:00/15:59) nach Intraday-Pattern-Confirmation
- 7 Patterns: BASELINE, VWAP_ABOVE, RECOVERY, STRONG_POS, POS_DAY, FAILED_FADE, TREND_DAY
- **Fixed-Hold: KEIN Edge** (0 DA Survivors, 468 Hypothesen)
- **Trailing Stop (SL=0.25 ADR, TRAIL_D, MaxDays=5): ALLE 21 Combos OOS ROBUST**
- Bester OOS: POS_DAY|15:00 EV=+0.458R, CI=[+0.27,+0.76], N=2719
- TREND_DAY|14:30: EV=+0.449R, CI=[+0.32,+0.59], Sharpe=+0.175, N=1425
- Sogar BASELINE|15:59: EV=+0.414R, CI=[+0.34,+0.49], N=5902
- WR niedrig (~28-40%), aber PF=1.5-1.8x (asymmetrisch: Winners ~+1.5R, Losers ~-0.2R)
- SL=0.50 ADR auch robust aber schwaecher (~+0.10R bis +0.22R)
- **Kern-Erkenntnis: Edge kommt hauptsaechlich vom Trailing-Stop auf Gap-Aktien**

### Ansatz 2: Pre-10am Momentum = OOS-BESTAETIGT
**Top: GapUp, gap>=2.0, od>=0.5, rvol>=3, RSI>=50, MomConfirm**
- Entry: close_935, SL: 0.15 ADR, TRAIL_D
- OOS: N=127, EV=+0.90R, WR=63.8%, P=0.0000, CI=[0.59,1.21]

### Ansatz 3: Quant Free Exploration = 8 OOS-BESTAETIGT
Top (alle GapUp, OD>0.5, TRAIL_D, SL=0.25 ADR):
| # | Strategie | OOS EV | OOS N | OOS p |
|---|-----------|--------|-------|-------|
| 1 | Earnings+OD>0.5+RVOL>5 | +0.80R | 58 | 0.0006 |
| 2 | OD>0.8+RVOL>5+SPY_pos | +0.73R | 85 | <0.0001 |
| 3 | BigFirstCandle>0.2 | +0.64R | 250 | <0.0001 |
| 4 | LowWick<0.15 | +0.45R | 253 | <0.0001 |
| 5 | BigBody>0.6 | +0.40R | 262 | <0.0001 |
Dip&Rip: IS +0.275R -> OOS +0.408R, WR=68.3%, N=293 (OOS ROBUST, CI=[+0.261,+0.572])
  Subfilter: OD>=0.8+RVOL>=5 OOS +0.814R N=21, Gap>=7% OOS +0.673R N=83

### Ansatz 4: Discretionary-Led (d13_discretionary_led.py)
6 Ideen getestet, 175 Hypothesen, Bonferroni Alpha=0.000286

**Idee A: Sell-Off & Recovery** = OOS-BESTAETIGT
- Narrativ: GapUp/Dn -> Sell-Off -> 50%/VWAP Recovery -> Long/Short
- Beste Variante: VWAP Reclaim, d=0.10, Selloff bis 10:30, Deadline 11:00
  - IS: N=2366, EV=+0.469R, WR=77.1%, p<0.0001
  - OOS: N=3128, EV=+0.438R, WR=74.3%, CI=[+0.406,+0.469] ROBUST (-7% Degradation)
- Subfilter OD>=0.8: OOS N=116, EV=+0.407R, CI=[+0.258,+0.557] ROBUST
- Subfilter RVOL:7++OD:0.8+: OOS N=39, EV=+0.530R, CI=[+0.284,+0.806] ROBUST

**Idee C: Failed Fade** = OOS-BESTAETIGT (SEHR STARK)
- Narrativ: Gap + OD gegen Gap + Gap NOT filled = extrem stark
- Baseline: IS N=1413, EV=+0.659R, WR=89.5%, p<0.0001
- OOS: N=1702, EV=+0.662R, WR=89.1%, CI=[+0.629,+0.694] ROBUST
- Subfilter OD>=0.8: OOS N=132, EV=+0.612R, CI=[+0.488,+0.731] ROBUST

**Verworfen:**
- Idee B (VWAP Reclaim): EV=-0.174R (negativ)
- Idee D (Range Compression): N=20 (zu wenig)
- Idee E (VPOC Bounce): EV=-0.072R (negativ)
- Idee F (Day+1 Follow-Through): EV=+0.072R, p=0.14 (zu schwach, failed DA)

## User-Vorgaben
- Minimum EV: 0.25R | Ideal: +0.5 ADR (+2.0R)
- OD darf als Variable genutzt werden
- IS/OOS strikt trennen

## Offene Aufgaben
1. ~~S4 Optimierung~~ DONE
2. ~~Ansatz 1~~ DONE (kein Edge)
3. ~~Ansatz 2~~ DONE
4. ~~Ansatz 3~~ DONE (8 OOS-bestaetigt)
5. ~~Follow-Up-Download~~ DONE (62,353 + VWAP + VP)
6. ~~Ansatz 1 Neuberechnung~~ DONE (kein Edge)
7. ~~Dip&Rip OOS-Validierung~~ DONE (ROBUST, +0.408R)
8. README-Update
9. Doku: D13_STRATEGIE_DOKUMENTATION.txt (V2, ~1100 Zeilen)
10. ~~Discretionary-Led Exploration~~ DONE (2 neue robuste Strategien: A + C)
11. ~~Ansatz 5: Late-Day Pattern-Confirmed Swing~~ DONE (TRAIL_D ROBUST, Fixed-Hold kein Edge)

## Polygon API
- Plan: 5 Jahre ab ~2021-02-22
- Script: scripts/06_download_followup_days.py
- Mapping: data/metadata/followup_day_mapping.parquet

## Was NICHT funktioniert
- Late-Day Swing BLIND (ohne Pattern): Kein Edge mit echten Daten
- Late-Day Fixed-Hold (mit Pattern): Kein DA Survivor (468 Hypothesen)
- VWAP Sigma-2 Mean Reversion: Negativ
- ORB Breakout: Negativ
- Late-Session 13:00: Negativ
- No-Fill Power Move: Artefakt
- VPOC Migration: Zu schwach
- Volume-Burst: -0.78R
- GapDn-Overnight/Swing: Negativ
- Reversal After Exhaustion: -0.21R
- Sector Momentum: Kein Effekt
- RSI Contrarian: -0.27R
- VWAP Reclaim (Idee B): EV=-0.174R
- Range Compression Breakout (Idee D): N=20 (zu selten)
- VPOC Bounce (Idee E): EV=-0.072R
- Day+1 Follow-Through (Idee F): EV=+0.072R (nicht signifikant)
