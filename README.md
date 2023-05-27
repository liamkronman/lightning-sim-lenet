# Lightning Sim on LeNet-300-100

## Overview
Event-driven simulation occurs in two phases:

1. Scheduling requests. For example, `schedule_lenet_requests()` does this by scheduling a particular number of requests with an even "interarrival" time between each's initiation.

2. Simulating the schedule. `Simulator.simulate()` will act out the requests at their specified times and return their average completion time.

## Usage
Run ```python3 lenet_plot.py``` to generate plot of average request completion times over different interarrivals lengths.