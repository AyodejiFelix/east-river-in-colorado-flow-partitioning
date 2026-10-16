# east-river-in-colorado-flow-partitioning
Tracer-informed physics-informed neural network (PINN) for fast/slow flow partitioning in the snow-dominated East River Watershed, Colorado, with LSTM, GRU, TCN and LightGBM baselines and blocked temporal validation.

This repository contains the code and results for “Tracer-Informed Physics-Informed Neural Networks for Partitioning Fast and Slow Flow Pathways in a Snow-Dominated Mountain Watershed”, prepared for submission to Water Resources Research.

The model is a physics-informed neural network that enforces joint water and chloride mass conservation across five nested sub-basins of the East River Watershed, Colorado (Rustlers Gulch, Copper Creek, East above Quigley, Middle East River, and the Pumphouse outlet). It uses exact closed-form integration of a two-reservoir (fast and slow) structure with routing through the sub-basin network. A hydrology-only ablation, with the chloride loss removed, isolates the contribution of the tracer from that of the water-balance structure.

The study compares the PINN against unconstrained LSTM, GRU, temporal-convolution and gradient-boosting baselines, plus two no-model floors (linear interpolation and day-of-year climatology). It reports several evaluation findings:

The discharge record at one gauge (Gothic_ME) fails a mass balance and is masked from the primary comparison.
A random held-out split rewards interpolation: linear interpolation matches the LSTM at four of five gauges.
Under blocked temporal validation, in which whole six-month blocks are withheld, both PINN variants outperform every data-driven baseline at the watershed outlet and at EAQ. The advantage comes from the water-balance structure, not the tracer.
Simulated baseflow fraction stays above an independent literature benchmark in every configuration tested.

The repository includes the source code, training and diagnostic scripts, the results needed to reproduce each figure and table, and documentation of the model architecture and findings. Raw input data are not redistributed; the README lists the ESS-DIVE and NRCS dataset DOIs and links to download them.

Data: ESS-DIVE (Watershed Function Science Focus Area) and USDA NRCS SNOTEL. Archived release: [Zenodo DOI to be added]. License: MIT.
