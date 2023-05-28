from lenet_sim import Simulator, schedule_lenet_requests
import matplotlib.pyplot as plt
from tqdm import tqdm

DATAPATH_LATENCY = 344
OVERHEAD_FACTOR = 0.1

fig, ax = plt.subplots()

def run_lenet_batch(label, dpl=0, overhead_factor=0.):
    '''
    Simulates and plots batches of LeNet requests under conditions

    Parameters
    ----------
    dpl: datapath latency for all LeNet requests in batch (in ts)
    overhead_factor: latency factor before all layers of LeNet requests in batch
    '''
    print(f"Running LeNet batches with {label}...")
    x_axis = []
    y_axis = []
    num_reqs = 100 # number of back-to-back lenet computations
    for interarrival_space in tqdm(range(0, 2000, 50)):
        simulator = Simulator()
        schedule_lenet_requests(simulator, num_reqs, interarrival_space, dpl, overhead_factor)
        average_job_time = simulator.simulate()
        # print(f'Average job time for interarrival space of {interarrival_space}: {average_job_time}')
        x_axis.append(interarrival_space)
        y_axis.append(average_job_time)

    ax.plot(x_axis, y_axis, label=label)

run_lenet_batch("perfect conditions")
run_lenet_batch("344ts datapath latency", dpl=DATAPATH_LATENCY)
run_lenet_batch("344ts DPL + overhead", dpl=DATAPATH_LATENCY, overhead_factor=OVERHEAD_FACTOR)

ax.set_xlabel('Time between Request arrivals (in ts)')
ax.set_ylabel('Average Request completion (in ts)')
ax.set_title('LeNet-300-100 on 300 cores over 100 requests')
ax.legend()
plt.yscale("log")
out_filepath = "sim_plots/avg_req_completion_vs_interarrival.png"
plt.savefig(out_filepath)
print(f"Output accessible in {out_filepath}")