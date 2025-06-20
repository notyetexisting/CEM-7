CEM-7: A Population Simulation Model

CEM-7 simulates humanity's population over 200 years under a hypothetical stress scenario: a rapid cultural shift where natural reproduction declines. It models the interplay between this shift, the rise of Assisted Reproductive Technologies (ART), and critical economic/social feedback loops.

The model's key feature is its dynamic feedback system:

Economic Cascade: Population decline weakens the economy, which in turn suppresses the rollout of expensive ART.

Social Aversion: A visible demographic crisis can slow the underlying cultural shift.

This allows the model to explore complex, non-linear "death spiral" or "stabilization" scenarios. ## Quickstart

1. Installation
pip install numpy pandas matplotlib statsmodels

2. Run a Default Simulation

Execute the script directly to run the baseline scenario (500 simulations):

python cem7.py


This will run the model and generate a multi-panel plot visualizing the results.

Customizing a Simulation

The most efficient way to use the model is to modify the SimulationConfig object inside the if __name__ == "__main__": block of cem7.py.

This lets you test different hypotheses without altering the core simulation logic.

Example: "Techno-Optimist" Scenario

Edit the main block to create a new config:

if __name__ == "__main__":
    # Create a custom config for this run
    techno_optimist_config = SimulationConfig(
        base_compliance=0.95,      # Higher ART adoption
        h_midpoint_year=100        # Slower cultural shift
    )
    
    # Initialize the model with the custom config and run
    model = DemographicModel(techno_optimist_config, AIForecaster())
    results_df = model.run_simulation()
    plot_results(results_df, techno_optimist_config)

Key Parameters to Modify

base_compliance: How readily couples adopt ART (0.0 to 1.0).

h_midpoint_year: The speed of the cultural shift (a higher number is a slower shift).

economic_decline_rate: How sensitive the economy is to population loss.

crisis_aversion_strength: How strongly society reacts to a visible population crisis.

Interpreting the Output

Population Plot: The main result. Shows the median trajectory and the 10th-90th percentile uncertainty range.

Subplots: Use the other plots (Effective Tech, Economic Health, etc.) to understand the drivers of the population trend. A sharp drop in Effective Tech indicates an economic cascade was a primary factor.
