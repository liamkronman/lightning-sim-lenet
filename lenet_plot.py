from lenet_sim import Simulator, schedule_lenet_requests
import matplotlib.pyplot as plt
from tqdm import tqdm

num_reqs = 100 # number of back-to-back lenet computations
for interarrival_space in tqdm(range(0, 2000, 100)):
    simulator = Simulator()
    schedule_lenet_requests(simulator, 100, interarrival_space)
    average_job_time = simulator.simulate()
    # print(f'Average job time for interarrival space of {interarrival_space}: {average_job_time}')
    plt.plot(interarrival_space, average_job_time, "-o")

plt.xlabel('Time between Request arrivals (in ts)')
plt.ylabel('Average Request completion (in ts)')
plt.title('LeNet-300-100 on 300 cores over 100 requests (344ts DPL + overhead)')
out_filepath = "sim_plots/avg_req_completion_vs_interarrival_344_dpl_and_overhead.png"
plt.savefig(out_filepath)
print(f"Output accessible in {out_filepath}")