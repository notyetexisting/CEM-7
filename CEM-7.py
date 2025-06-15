import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from dataclasses import dataclass
from typing import List

# -------------------------------------------------------------------
# CEM-7: AI-Enhanced Extinction Model with Dynamic Feedback
# -------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Central configuration for the simulation scenario."""
    # World Anchors
    initial_population: float = 8.1e9
    years: int = 200
    extinction_threshold: float = 1e6

    # Demographic Rates
    base_birth_rate: float = 0.017  # More realistic TFR baseline
    base_death_rate: float = 0.0075
    
    # Homosexuality Trend Parameters (Mean values for Monte Carlo)
    h_midpoint_year: float = 80  # t_H: Year at which H(t) is 0.5
    h_growth_rate: float = 0.06   # k_H: Steepness of the shift
    
    # Technology and Social Factors
    base_compliance: float = 0.75  # Fraction of couples attempting ART
    socio_political_decline_rate: float = 0.003 # Barriers to ART
    
    # Dynamic Feedback Loop Parameters
    crisis_aversion_threshold: float = 0.7 # Population fraction that triggers feedback
    crisis_aversion_strength: float = 0.5   # How much H(t) growth slows
    economic_decline_rate: float = 0.05    # How much economy is tied to pop growth
    
    # Simulation Settings
    n_simulations: int = 500


class AIForecaster:
    """Manages AI-driven forecasting for technology adoption."""
    def __init__(self):
        # More realistic historical data for a better forecast
        historical_tech = np.array([
            0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.15, 0.19, 0.24, 
            0.30, 0.36, 0.42, 0.48, 0.54, 0.60
        ])
        
        # Using Holt-Winters as a stand-in for a more complex AI model
        model = ExponentialSmoothing(historical_tech, trend='add', seasonal=None, damped_trend=True)
        fit = model.fit()
        
        # Forecast for the entire simulation period
        self.forecast = np.clip(fit.forecast(SimulationConfig.years), 0, 1.0)

    def get_tech_level(self, year: int) -> float:
        """Returns the forecasted technology level for a given year."""
        return self.forecast[year - 1] if year > 0 else 0.0


class DemographicModel:
    """Encapsulates the core logic of the population simulation."""
    def __init__(self, config: SimulationConfig, tech_forecaster: AIForecaster):
        self.config = config
        self.tech_forecaster = tech_forecaster

        # Historic fertility decline data (from paper context)
        # Assumes TFR drops from ~2.4 to ~1.7 over the period
        self.fertility_decline_factor = np.linspace(1.0, 1.7 / 2.4, config.years)

    def run_simulation(self) -> pd.DataFrame:
        """Runs a full Monte Carlo simulation and returns results."""
        all_results = []

        for _ in range(self.config.n_simulations):
            # --- Initialize state for this single simulation run ---
            # Introduce stochasticity as per CEM-1
            sim_params = {
                't_H': random.gauss(self.config.h_midpoint_year, 10),
                'k_H': random.uniform(self.config.h_growth_rate - 0.02, self.config.h_growth_rate + 0.02),
                'compliance': random.uniform(self.config.base_compliance - 0.15, self.config.base_compliance + 0.15),
                'sp_decline': random.uniform(0.001, 0.005)
            }
            
            P = self.config.initial_population
            peak_population = P
            economic_health = 1.0
            
            # Store results for this run
            history = {
                'population': [], 'h_fraction': [], 't_effective': [], 'economic_health': []
            }

            for t in range(1, self.config.years + 1):
                # --- Calculate intermediate variables for year t ---
                
                # Dynamic Feedback 1: Crisis Aversion
                current_k_H = sim_params['k_H']
                if P < self.config.crisis_aversion_threshold * peak_population:
                    current_k_H *= self.config.crisis_aversion_strength

                H_t = 1 / (1 + np.exp(-current_k_H * (t - sim_params['t_H'])))
                
                # Dynamic Feedback 2: Economic Health
                pop_growth_rate = (P - history['population'][-1]) / history['population'][-1] if t > 1 else 0
                if pop_growth_rate < 0:
                    economic_health = max(0.1, economic_health + pop_growth_rate * self.config.economic_decline_rate)
                
                T_forecasted = self.tech_forecaster.get_tech_level(t)
                T_effective = T_forecasted * economic_health # Economy impacts tech deployment
                
                # Social/Political Barriers
                S_t = max(0, 1 - sim_params['sp_decline'] * t)
                
                # --- Update Population ---
                fertility_modifier = self.fertility_decline_factor[t - 1]
                
                birth_rate_natural = self.config.base_birth_rate * fertility_modifier * (1 - H_t)
                birth_rate_art = self.config.base_birth_rate * fertility_modifier * (H_t * T_effective * sim_params['compliance'] * S_t)
                
                births = (birth_rate_natural + birth_rate_art) * P
                deaths = self.config.base_death_rate * P
                P += (births - deaths)

                # --- Record and check for exit conditions ---
                peak_population = max(peak_population, P)
                history['population'].append(P)
                history['h_fraction'].append(H_t)
                history['t_effective'].append(T_effective)
                history['economic_health'].append(economic_health)

                if P < self.config.extinction_threshold:
                    break
            
            all_results.append(pd.DataFrame(history))
        
        # --- Process and aggregate results ---
        # Pad shorter simulations with their last value
        max_len = max(len(df) for df in all_results)
        for df in all_results:
            if len(df) < max_len:
                last_row = df.iloc[-1]
                padding = pd.DataFrame([last_row] * (max_len - len(df)))
                df = pd.concat([df, padding], ignore_index=True)
            df.index = range(1, len(df) + 1) # Ensure common index
            
        # Combine all dataframes for easy quantile calculation
        combined_df = pd.concat(all_results)
        summary_df = combined_df.groupby(combined_df.index).quantile([0.1, 0.5, 0.9]).unstack()
        
        return summary_df

def plot_results(df: pd.DataFrame, config: SimulationConfig):
    """Generates a multi-panel plot of the simulation results."""
    fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    years_axis = df.index

    # --- Panel 1: Population Trajectory ---
    axs[0].plot(years_axis, df[('population', 0.5)] / 1e9, color='blue', label='Median Population')
    axs[0].fill_between(years_axis, df[('population', 0.1)] / 1e9, df[('population', 0.9)] / 1e9, 
                        color='blue', alpha=0.2, label='10th-90th Percentile')
    axs[0].axhline(config.initial_population / 1e9, linestyle='--', color='green', label='Initial Population (2025)')
    axs[0].set_title('CEM-7: Population Trajectory under Dynamic Feedback', fontsize=16)
    axs[0].set_ylabel('Population (Billions)')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # --- Panel 2: Homosexuality Fraction (H) ---
    axs[1].plot(years_axis, df[('h_fraction', 0.5)], color='red', label='Median H(t)')
    axs[1].fill_between(years_axis, df[('h_fraction', 0.1)], df[('h_fraction', 0.9)],
                        color='red', alpha=0.2)
    axs[1].set_title('Fraction of Population Identifying as Homosexual (H)', fontsize=14)
    axs[1].set_ylabel('Fraction (0.0 to 1.0)')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # --- Panel 3: Effective Technology Level (T_eff) ---
    axs[2].plot(years_axis, df[('t_effective', 0.5)], color='purple', label='Median Effective Tech')
    axs[2].fill_between(years_axis, df[('t_effective', 0.1)], df[('t_effective', 0.9)],
                        color='purple', alpha=0.2)
    axs[2].set_title('Effective ART Level (T_eff = T_forecast * Economic Health)', fontsize=14)
    axs[2].set_ylabel('Efficacy (0.0 to 1.0)')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.6)
    
    # --- Panel 4: Economic Health ---
    axs[3].plot(years_axis, df[('economic_health', 0.5)], color='orange', label='Median Economic Health')
    axs[3].fill_between(years_axis, df[('economic_health', 0.1)], df[('economic_health', 0.9)],
                        color='orange', alpha=0.2)
    axs[3].set_title('Economic Health Factor', fontsize=14)
    axs[3].set_ylabel('Factor (0.0 to 1.0)')
    axs[3].set_xlabel('Years from 2025')
    axs[3].legend()
    axs[3].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Initialize components
    config = SimulationConfig()
    ai_forecaster = AIForecaster()
    model = DemographicModel(config, ai_forecaster)
    
    # Run the simulation
    print("Running CEM-7 Monte Carlo simulation...")
    results_df = model.run_simulation()
    print("Simulation complete. Generating plots...")
    
    # Visualize the results
    plot_results(results_df, config)