import math

NUM_CORES = 300
LENET_LAYERS = [(784, 300), (300, 100), (100, 10)]

class Simulator():
    def __init__(self):
        self.cores = [Core() for _ in range(NUM_CORES)]
        self.next_core = 0

    def simulate_lenet(self):
        job_times = []

        for layer in LENET_LAYERS:
            input_size, vvps = layer
            earliest_start = math.inf
            latest_end = 0

            for _ in range(vvps):
                vvp_start_time, vvp_end_time = self.cores[self.next_core % NUM_CORES].schedule_vvp(input_size)
                if vvp_start_time < earliest_start:
                    earliest_start = vvp_start_time
                if vvp_end_time > latest_end:
                    latest_end = vvp_end_time
                self.next_core += 1
            
            job_time = latest_end - earliest_start
            job_times.append(job_time)
        
        return job_times

class Core():
    def __init__(self):
        self.queue_total_time = 0

    def schedule_vvp(self, input_size):    
        start_time = self.queue_total_time
        end_time = start_time + input_size
        self.queue_total_time += input_size
        
        return start_time, end_time
    
if __name__ == "__main__":
    simulator = Simulator()
    