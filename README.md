Event-driven simulation occurs in two phases:

1. Scheduling requests. For example, schedule_lenet_requests() does this by scheduling a particular number of requests with an even "interarrival" time between each's initiation.

2. Simulating the schedule. Simulator.simulate() will act out the requests at their specified times and return their average completion time.