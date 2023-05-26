from lenet_sim import Simulator, simulate_lenet_requests
import matplotlib.pyplot as plt

num_reqs = 100 # number of back-to-back lenet computations
for interarrival_space in range(200, 2000, 100):
    simulator = Simulator()
    average_job_time = simulate_lenet_requests(simulator, 100, interarrival_space)
    print(f'Average job time for interarrival space of {interarrival_space}: {average_job_time}')
    plt.plot(interarrival_space, average_job_time, "-o")

plt.xlabel('Time between Request arrivals (in ts)')
plt.ylabel('Average Job completion (in ts)')
plt.title('LeNet-300-100 on 300 cores over 100 requests (perfect conditions)')
plt.savefig("sim_plots/avg_job_completion_vs_interarrival_perfect_conditions.png")