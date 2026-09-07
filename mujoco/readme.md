A mujoco simulation

Apply forces at the point where the drone is placed. Measure the reaction force generated at the location of the rig where the load cell will be placed 
experimentally. No drones should be placed at this moment, just the force time series should be provided. The force hence evlauated at the position of load cell,
which is the pivot, should be experimentally validated, from the load cell readings (Force time series).

To do:

1. Align the load cell, and the wrench location using CAD

2. Safe sets - for reverse mapping the cusps to the causal forces

3. Adding gust/noise

4. Exact same updations to gimbal+drone setup code to map control input to forces at the load cell

5. Cross validation by pre-multiplying F_{sensor} with the \mathcal{C}_{AN} transformation matrix, to get F_{applied}