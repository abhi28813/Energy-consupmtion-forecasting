'''
Tariff-based load-shifting cost optimizer.

Takes the plant's historical (or model-forecasted) hourly energy usage,
applies a Time-of-Use (ToU) industrial tariff, and simulates shifting a
configurable share of peak-window load into cheaper off-peak hours to
estimate the potential reduction in peak-tariff energy cost.

Outputs (under artifacts/) are plain CSV/JSON so they can be dropped
straight into a Power BI dashboard:
  - load_shift_report.csv     hourly usage/cost before vs after shifting
  - load_shift_scenarios.csv  savings % at several shift-fraction scenarios
  - load_shift_summary.json   headline numbers for the default scenario
'''
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Tuple

import pandas as pd

from src.exception import CustomException
from src.logger import logging


@dataclass
class TariffConfig:
    '''
    Time-of-Use (ToU) tariff card. Rates are per kWh in a single currency
    unit — replace with the plant's actual utility tariff card. Hour
    windows follow a typical industrial ToD structure: morning + evening
    grid peaks, cheap night off-peak, standard rate the rest of the day.
    '''
    peak_rate: float = 10.5
    standard_rate: float = 7.0
    off_peak_rate: float = 4.5

    peak_hours: Tuple[int, ...] = (9, 10, 11, 18, 19, 20)
    off_peak_hours: Tuple[int, ...] = (22, 23, 0, 1, 2, 3, 4, 5, 6)
    # every other hour of the day is billed at standard_rate

    def rate_for_hour(self, hour: int) -> float:
        if hour in self.peak_hours:
            return self.peak_rate
        if hour in self.off_peak_hours:
            return self.off_peak_rate
        return self.standard_rate

    def period_for_hour(self, hour: int) -> str:
        if hour in self.peak_hours:
            return 'Peak'
        if hour in self.off_peak_hours:
            return 'Off-Peak'
        return 'Standard'


@dataclass
class LoadShiftConfig:
    # Share of PEAK-window load assumed shiftable to off-peak hours
    # (e.g. batch material handling / non-continuous furnace operations).
    # Continuous-process load is assumed non-shiftable.
    shiftable_fraction: float = 0.40
    scenario_fractions: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)

    output_path: str = os.path.join('artifacts', 'load_shift_report.csv')
    scenarios_path: str = os.path.join('artifacts', 'load_shift_scenarios.csv')
    summary_path: str = os.path.join('artifacts', 'load_shift_summary.json')


class TariffLoadShiftOptimizer:
    def __init__(self, tariff: TariffConfig = None, shift_config: LoadShiftConfig = None):
        self.tariff = tariff or TariffConfig()
        self.shift_config = shift_config or LoadShiftConfig()

    def _hourly_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Aggregates raw readings into total kWh per hour-of-day (0-23).'''
        try:
            data = df.copy()
            data['hour'] = data['hours'].astype(int) % 24

            profile = data.groupby('hour')['Usage_kWh'].sum()
            profile = profile.reindex(range(24), fill_value=0.0).reset_index()
            profile.columns = ['hour', 'Usage_kWh']
            profile['period'] = profile['hour'].apply(self.tariff.period_for_hour)
            profile['rate_per_kWh'] = profile['hour'].apply(self.tariff.rate_for_hour)
            return profile
        except Exception as e:
            raise CustomException(e, sys)

    def simulate(self, df: pd.DataFrame, shiftable_fraction: float = None):
        '''
        Shifts `shiftable_fraction` of energy out of each peak hour and
        redistributes it evenly across off-peak hours, then compares the
        tariff cost before vs after. Total energy consumed is conserved —
        only its timing changes.
        '''
        try:
            f = self.shift_config.shiftable_fraction if shiftable_fraction is None else shiftable_fraction
            profile = self._hourly_profile(df)

            peak_mask = profile['period'] == 'Peak'
            off_peak_mask = profile['period'] == 'Off-Peak'
            if off_peak_mask.sum() == 0:
                raise ValueError('No off-peak hours configured to absorb shifted load')

            original_cost = profile['Usage_kWh'] * profile['rate_per_kWh']

            shifted_usage = profile['Usage_kWh'].copy()
            shifted_out_total = (shifted_usage[peak_mask] * f).sum()
            shifted_usage[peak_mask] *= (1 - f)
            shifted_usage[off_peak_mask] += shifted_out_total / off_peak_mask.sum()

            optimized_cost = shifted_usage * profile['rate_per_kWh']

            report = pd.DataFrame({
                'hour': profile['hour'],
                'period': profile['period'],
                'rate_per_kWh': profile['rate_per_kWh'],
                'original_kWh': profile['Usage_kWh'],
                'shifted_kWh': shifted_usage,
                'original_cost': original_cost,
                'optimized_cost': optimized_cost,
            })

            total_before = report['original_cost'].sum()
            total_after = report['optimized_cost'].sum()
            savings_pct = (total_before - total_after) / total_before * 100

            summary = {
                'shiftable_fraction': f,
                'total_energy_kWh': float(report['original_kWh'].sum()),
                'shifted_energy_kWh': float(shifted_out_total),
                'cost_before': float(total_before),
                'cost_after': float(total_after),
                'savings_amount': float(total_before - total_after),
                'savings_pct': float(savings_pct),
            }

            logging.info(f'Load-shift simulation ({f:.0%} of peak load shifted): '
                         f'{savings_pct:.2f}% projected cost reduction')

            return report, summary
        except Exception as e:
            logging.info('Exception occured during load-shift simulation')
            raise CustomException(e, sys)

    def run(self, data_path: str = os.path.join('artifacts', 'data.csv')) -> dict:
        try:
            df = pd.read_csv(data_path)

            report, summary = self.simulate(df)
            os.makedirs(os.path.dirname(self.shift_config.output_path), exist_ok=True)
            report.to_csv(self.shift_config.output_path, index=False)

            scenarios = [self.simulate(df, shiftable_fraction=f)[1]
                         for f in self.shift_config.scenario_fractions]
            pd.DataFrame(scenarios).to_csv(self.shift_config.scenarios_path, index=False)

            with open(self.shift_config.summary_path, 'w') as fobj:
                json.dump({'default_scenario': summary, 'scenarios': scenarios}, fobj, indent=2)

            logging.info(f'Load-shift report saved to {self.shift_config.output_path}')
            logging.info(f'Load-shift scenarios saved to {self.shift_config.scenarios_path}')

            print(f"Hourly load-shift report   : {self.shift_config.output_path}")
            print(f"Scenario sensitivity table : {self.shift_config.scenarios_path}")
            print(f"Default scenario ({summary['shiftable_fraction']:.0%} of peak load shifted to off-peak): "
                  f"{summary['savings_pct']:.2f}% projected reduction in peak-tariff energy cost")

            return summary
        except Exception as e:
            logging.info('Exception occured while running load-shift optimizer')
            raise CustomException(e, sys)


if __name__ == '__main__':
    optimizer = TariffLoadShiftOptimizer()
    optimizer.run()
