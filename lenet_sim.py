from sim_classes import Request, Job, Event, Task, LayerProgress
from collections import deque
from heapq import merge
from typing import List, Dict, Set, Callable, Tuple
import math

NUM_CORES = 300
LENET_LAYERS = [(784, 300), (300, 100), (100, 10)]

class Simulator():
    def __init__(self, dpl=0, overhead_factor=0.) -> None:
        self.dpl = dpl                                          # datapath latency (in ts) before every request
        self.overhead_factor = overhead_factor                  # latency factor between layers of request (proportional to input size of next layer)
        self.cores = [Core(i) for i in range(NUM_CORES)]        # cores that the simulator will run on
        self.next_core = 0                                      # core that follows the last scheduled core (round-robin style)
        self.queue:deque = deque([])                            # holds all scheduled events
        self.time = 0                                           # simulator's internal time
        self.req_id = 0                                         # id that will be assigned to the next Request
        self.req_start_times:Dict[int,int] = {}                 # maps req_ids to their request's scheduled start times
        self.req_end_times:Dict[int,int] = {}                   # maps req_ids to latest end times of any VVPs associated with that Request
        self.reqs_in_progress:Set[int] = set()                  # ids of all requests being processed at current time
        self.req_layer_progress:Dict[int,LayerProgress] = {}    # maps req_ids to LayerProgress(num_vvps_left, dependent_layers) for requests in progress
        self.req_times:List[int] = []                           # completion times of finished requests (at current time)

    def schedule_lenet(self, layers:List[Tuple[int, int]], start_t:int) -> None:
        '''
        Parameters
        ----------
        start_t: time (in ts of simulator) of request's arrival
        layers: list of tuples that outline the input size and number of VVPs per layer
        '''
        self.merge_into_queue([Request(start_t, layers, self.req_id)])
        self.req_id += 1

    def merge_into_queue(self, events:List[Event]) -> None:
        '''
        Parameters
        ----------
        events: list of events to merged into queue (by start_t)
        '''
        self.queue = deque(merge(events, self.queue, key=lambda event:event.start_t)) # added left-handedly to prioritize recents
    
    def update_req_layer_progress(self, req_id:int) -> None:
        '''
        Decrements the number of VVPs left to complete a layer in DNN and schedules next layer, if exists,
        otherwise logs total time the request took to process.

        Parameters
        ----------
        req_id: id of the request whose layer we just finished a computation on
        '''
        layer_progress = self.req_layer_progress[req_id]
        num_vvps_left, dependent_layers = layer_progress.num_vvps_left, layer_progress.dependent_layers
        if req_id not in self.req_end_times or self.time > self.req_end_times[req_id]:
            self.req_end_times[req_id] = self.time # setting endtime of request to latest endtime of a VVP
        if num_vvps_left == 1:
            if dependent_layers: # when there are still children layers
                input_size, vvps = dependent_layers[0] # for next layer
                overhead_time = math.ceil(self.overhead_factor * input_size)
                next_job = Job(self.time + overhead_time, req_id, vvps, input_size)
                self.merge_into_queue([next_job])
                self.req_layer_progress[req_id] = LayerProgress(vvps, dependent_layers[1:]) # to prevent aliasing
                self.time -= 1 # to keep it at same time on next cycle (so we don't skip the job we just scheduled)
            else: # request done
                total_req_time = max(self.time, self.req_end_times[req_id]) - self.req_start_times[req_id]
                self.req_times.append(total_req_time)
                self.reqs_in_progress.remove(req_id)
                # might want to remove from self.req_layer_progress[req_id]
        self.req_layer_progress[req_id].num_vvps_left -= 1

    def simulate(self) -> float:
        '''
        Performs simulation based on scheduled requests

        Returns
        -------
        average_request_time: average lifetime (in ts) of a request in simulation
        '''
        while self.queue or self.reqs_in_progress:
            while self.queue and self.queue[0].start_t == self.time: # there is an event that beginning now
                event = self.queue.popleft()
                if isinstance(event, Request):
                    self.req_start_times[event.req_id] = self.time
                    first_job, dependent_layers = event.gen_job_dag(self.time + self.dpl) # first layer of DAG representing that DNN and subsequent layers
                    self.merge_into_queue([first_job])
                    self.reqs_in_progress.add(event.req_id)
                    self.req_layer_progress[event.req_id] = LayerProgress(first_job.vvps, dependent_layers)
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
    def __init__(self, core_id:int) -> None:
        self.core_id = core_id                                  # unique identifier for core
        self.wait_queue:deque = deque([])                       # holds all tasks scheduled to core that haven't started being processed
        self.current_task_end_time = None                       # end time of task currently being processed
        self.current_req_id = None                              # req_id of task currently being processed
    
    def schedule_vvp(self, task:Task) -> None:
        '''
        Adds a task to the wait_queue of core

        Parameters
        ----------
        task: a VVP that wants to be processed on this core
        '''
        self.wait_queue.append(task)

    def load_new_task(self, sim_time:int) -> None:
        '''
        Loads new task off queue

        Parameters
        ----------
        sim_time: simulator's time (used to generate end time of task)
        '''
        new_vvp = self.wait_queue.popleft()
        self.current_task_end_time = new_vvp.size + sim_time
        self.current_req_id = new_vvp.req_id

    def time_step(self, sim_time:int, update_req_layer_progress:Callable[[int], None]) -> None:
        '''
        Simulates time step for core

        Parameters
        ----------
        sim_time: simulator's time
        update_req_layer_progress: callback that decrements number of tasks left for a request's layer to complete
        '''
        if not self.current_task_end_time and self.wait_queue: # core is unutilized
            self.load_new_task(sim_time)

        if sim_time == self.current_task_end_time and isinstance(self.current_req_id, int):
            update_req_layer_progress(self.current_req_id) # signal that the task has been complete
            if self.wait_queue:
                self.load_new_task(sim_time)
            else:
                # marks itself as unutilized
                self.current_task_end_time = None
                self.current_req_id = None


def schedule_lenet_requests(simulator, num_reqs:int, interarrival_space:int, dpl=0, overhead_factor=0.) -> None:
    '''
    Parameters
    ----------
    simulator: Simulator object we are scheduling requests onto
    num_reqs: number of requests to schedule
    interarrival_space: length of time (in ts) to stagger each request
    dpl: datapath latency for all LeNet requests (in ts)
    overhead_factor: latency factor before all layers of LeNet requests
    '''
    for req_id in range(num_reqs):
        simulator.schedule_lenet(LENET_LAYERS, req_id * interarrival_space)


if __name__=="__main__":
    simulator = Simulator()
    # schedule_lenet_requests(simulator, 100, 1200)

    # simulation of single lenet request
    simulator.schedule_lenet(LENET_LAYERS,0)
    print("Average job completion time (in ts):", simulator.simulate())