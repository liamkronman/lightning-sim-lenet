from sim_classes import Request, Job
from collections import deque
from heapq import merge

NUM_CORES = 300
LENET_LAYERS = [(784, 300), (300, 100), (100, 10)]

class Simulator():
    def __init__(self):
        self.cores = [Core(i) for i in range(NUM_CORES)]
        self.next_core = 0
        self.queue = deque([])
        self.req_id = 0 # the id that will be assigned to the next Request
        self.req_start_times = {}
        self.req_end_times = {}
        self.reqs_in_progress = set()
        self.time = 0
        self.req_layer_progress = {} # maps req_ids to [num_vvps_left, dependent_layers] for requests in progress
        self.req_times = []

    def schedule_lenet(self, layers, start_t=0):
        '''
        Parameters
        ----------
        start_t: time (in ts of simulator) of request's arrival
        layers: list of tuples that outline the input size and number of VVPs per layer
        '''
        self.merge_into_queue([Request(start_t, layers, self.req_id)])
        self.req_id += 1

    def merge_into_queue(self, events):
        '''
        Parameters
        ----------
        events: list of events to merged into queue (by start_t)
        '''
        self.queue = deque(merge(events, self.queue, key=lambda event:event.start_t)) # added left-handedly to prioritize recents
    
    def update_req_layer_progress(self, req_id):
        num_vvps_left, dependent_layers = self.req_layer_progress[req_id]
        if req_id not in self.req_end_times or self.time > self.req_end_times[req_id]:
            self.req_end_times[req_id] = self.time # setting to 
        if num_vvps_left == 1:
            if dependent_layers: # when there are still children layers
                input_size, vvps = dependent_layers[0]
                next_job = Job(self.time, req_id, vvps, input_size)
                self.merge_into_queue([next_job])
                self.req_layer_progress[req_id] = [vvps, dependent_layers[1:].copy()] # to prevent aliasing
                self.time -= 1 # to keep it at same time on next cycle
            else: # request done
                total_req_time = max(self.time, self.req_end_times[req_id]) - self.req_start_times[req_id]
                self.req_times.append(total_req_time)
                self.reqs_in_progress.remove(req_id)
                # might want to remove from self.req_layer_progress[req_id]
        self.req_layer_progress[req_id][0] -= 1

    def simulate(self):
        '''
        Performs simulation based on scheduled requests

        Returns
        -------
        average_request_time: average lifetime (in ts) of a request in simulation
        '''
        while self.queue or self.reqs_in_progress:
            while self.queue and self.queue[0].start_t == self.time:
                event = self.queue.popleft()
                if isinstance(event, Request):
                    self.req_start_times[event.req_id] = self.time
                    first_job, dependent_layers = event.gen_job_dag(self.time) # first layer of DAG representing that DNN and subsequent layers
                    self.merge_into_queue([first_job])
                    self.reqs_in_progress.add(event.req_id)
                    self.req_layer_progress[event.req_id] = [first_job.vvps, dependent_layers]
                elif isinstance(event, Job):
                    tasks = event.gen_tasks()
                    for task in tasks:
                        self.cores[self.next_core % NUM_CORES].schedule_vvp(task)
                        self.next_core += 1

            for core in self.cores:
                core.time_step(self.time, self.update_req_layer_progress)
            self.time += 1 # must come here (at end of loop) so as not to cause off-by-one error

        average_request_time = sum(self.req_times) / len(self.req_times)
        return average_request_time


class Core():
    def __init__(self, core_id):
        self.core_id = core_id
        self.time = 0
        self.wait_queue = deque([])
        self.current_task_end_time = None
        self.current_req_id = None # req_id of task currently being processed
    
    def schedule_vvp(self, task):
        self.wait_queue.append(task)

    def time_step(self, sim_time, update_req_layer_progress):
        '''
        Simulates time step for core

        Parameters
        ----------
        sim_time: simulator's time
        update_req_layer_progress: callback that decrements number of tasks left for a request's layer to complete
        '''
        # print("core #:", self.core_id)
        # print("time:", sim_time)
        self.time = sim_time # always inherit time of simulator

        # print("current_task_end_time",self.current_task_end_time)
        if not self.current_task_end_time and self.wait_queue:
            new_vvp = self.wait_queue.popleft()
            self.current_task_end_time = new_vvp.size + self.time
            self.current_req_id = new_vvp.req_id

        if self.time == self.current_task_end_time:
            update_req_layer_progress(self.current_req_id) # signal that the task has been complete
            if self.wait_queue:
                # load new task off queue
                new_vvp = self.wait_queue.popleft()
                self.current_task_end_time = new_vvp.size + self.time
                self.current_req_id = new_vvp.req_id
            else:
                self.current_task_end_time = None
                self.current_req_id = None


def schedule_lenet_requests(simulator, num_reqs, interarrival_space):
    for req_id in range(num_reqs):
        simulator.schedule_lenet(LENET_LAYERS, req_id * interarrival_space)


if __name__=="__main__":
    simulator = Simulator()
    schedule_lenet_requests(simulator, 100, 1200)

    # simulation of single lenet request
    # simulator.schedule_lenet(LENET_LAYERS,0)
    print(simulator.simulate())