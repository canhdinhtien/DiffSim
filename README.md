# Driver Tracking Time-Series Generation
## Overview
This repository is designed to **research, implement, and evaluate time-series generation methods**, focusing on data tracking driver activity in ride-hailing systems.
In the initial phase, the repository uses **simulated data** generated from predefined business rules and probability distributions as reference data in place of real production data.
Generative models are then trained to learn the statistical characteristics, temporal structure, and relationships between variables in the reference data, enabling them to generate new time series with similar properties.
## Objectives
The repository pursues the following main objectives:
1. **Implementation:** Implement and experiment with a range of time-series generation methods, from traditional statistical approaches to modern generative models such as VAE, GAN, Diffusion Models, and Flow Matching.
2. **Evaluation & Benchmarking:** Establish a process for evaluating and comparing methods on the same dataset, focusing on:
   - **Statistical Fidelity:** Similarity in distributions and statistical characteristics between the original and generated data.
   - **Temporal Fidelity:** The ability to preserve temporal structure, dependencies, and correlations between variables.
   - **Diversity:** Diversity of the generated data, limiting mode collapse or copying of training data.
   - **Practical Utility:** The usability of generated data for downstream tasks and its compliance with business constraints.
   - **Computational Efficiency:** Training cost, data generation speed, and scalability.
3. **Feasibility Study:** Assess the feasibility of using generative models to reproduce key properties of driver tracking data, while identifying the strengths, limitations, and conditions of use of each method.
4. **Production Readiness:** Select promising methods for further validation on real data, with the aim of deployment in production environments.
## Experimental Approach
The research process is divided into four stages:
1. **Data Simulation:** Build a simulated tracking dataset based on controlled assumptions, business rules, and probability distributions.
2. **Model Implementation:** Implement, train, and tune different data generation methods on the same reference dataset.
3. **Evaluation & Comparison:** Evaluate generated data quality using consistent metrics, conduct quantitative comparisons, and analyze the results.
4. **Real-world Validation:** Validate successful methods on real data once production data becomes available.
## Scope & Limitations
The reference data in the initial phase is simulated data, rather than data collected directly from actual driver activity.
Therefore, experiments on simulated data primarily aim to **assess the ability to learn and reproduce data properties, compare methods, and demonstrate technical feasibility**.
These results are not sufficient to establish that a model can accurately reproduce real-world driver behavior or meet production deployment requirements. Such conclusions require further validation on real data.
:::
A note on terminology: Use simulated real data or reference data rather than simply real data, because your data is generated from predefined rules and distributions rather than collected from a real system.
Another important distinction is between learning to reproduce the simulated distribution and generalizing to production data. These should be two separate evaluation stages to avoid overstating conclusions from benchmark results.