# Nowcasting Peruvian inflation with mixed-frequency and machine learning models

[![CI](https://github.com/Rodgrandez/peru-inflation-nowcasting/actions/workflows/ci.yml/badge.svg)](https://github.com/Rodgrandez/peru-inflation-nowcasting/actions/workflows/ci.yml)

How much does the estimate of the current month's inflation in Lima improve as daily data arrive during the
month, and which models use that information best? This project compares a random walk, an AR benchmark,
U-MIDAS and Almon-MIDAS regressions, Ridge, LASSO, XGBoost and a forecast combination, re-estimated every
month on an expanding window, for headline inflation and inflation excluding food and energy.

<!-- RESULTS:START -->
<!-- RESULTS:END -->

![Relative RMSE by week](reports/figures/rel_rmse_headline.png) ![Nowcast vs actual](reports/figures/nowcast_headline.png)

## Data (public, BCRP statistics API)
- Targets: monthly CPI inflation for Metropolitan Lima (PN01271PM) and CPI excluding food and energy (PN01276PM).
- Monthly: 12-month-ahead inflation expectations survey (PD12912AM), used with a one-month lag.
- Daily: interbank exchange rate (PD04638PD), interbank interest rate (PD04692MD), WTI oil (PD04705XD),
  wheat (PD31887XD), maize (PD31888XD) and soybean oil (PD31889XD).

## Design
- Information sets at the end of weeks 1-4 (days 7, 14, 21 and month end). Inflation and expectations for the
  current month are never used: they are published after the month ends.
- Estimation starts in 2003-01 (inflation-targeting regime); nowcasts are evaluated from 2015-01.
- Diebold-Mariano tests on squared errors with the Harvey-Leybourne-Newbold correction.

## Reproduce
```bash
conda env create -f environment.yml && conda activate inflation-nowcast
make all        # or: python -m nowcast.pipeline all
```

License: MIT.
