import math

NUM_CORES = 300
LENET_LAYERS = [(784, 300), (300, 100), (100, 10)]

class Simulator():
    def __init__(self):
        self.cores = [Core() for _ in range(NUM_CORES)]
        self.next_core = 0

    def simulate_lenet(self, t_offset=0):
        job_times = []

        for layer in LENET_LAYERS:
            input_size, vvps = layer
            earliest_start = math.inf
            latest_end = 0

            for _ in range(vvps):
                vvp_start_time, vvp_end_time = self.cores[self.next_core % NUM_CORES].schedule_vvp(input_size, t_offset)
                if vvp_start_time < earliest_start:
                    earliest_start = vvp_start_time
                if vvp_end_time > latest_end:
                    latest_end = vvp_end_time
                self.next_core += 1
            
            job_time = latest_end - earliest_start
            job_times.append(job_time)
            t_offset = latest_end # next layer must begin after this layer is complete
        
        total_time = sum(job_times)
        return total_time

class Core():
    def __init__(self):
        self.time = 0

    def schedule_vvp(self, input_size, t_offset=0):
        if t_offset > self.time: # when the vvp is scheduled for time after last vvp
            self.time = t_offset
        
        start_time = self.time
        end_time = start_time + input_size
        self.time += input_size
        
        return start_time, end_time
    
def simulate_lenet_requests(simulator, num_reqs, interarrival_space):
    req_times = []

    for req_id in range(num_reqs):
        req_times_for_req = simulator.simulate_lenet(req_id * interarrival_space)
        req_times.append(req_times_for_req)
    
    return sum(req_times) / len(req_times)