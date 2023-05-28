from typing import List, Tuple
from copy import deepcopy

class Event():
    def __init__(self, start_t:int, req_id:int) -> None:
        '''
        Parameters
        ----------
        start_t: time (in ts of simulator) of event's arrival (non-negative integer)
        req_id: identifier of request that the event is associated with (a natural number)
        '''
        self.start_t = start_t
        self.req_id = req_id


class Task():
    '''
    Represents VVP in a layer of DNN
    '''
    def __init__(self, req_id:int, size:int) -> None:
        self.req_id = req_id
        self.size = size


class Job(Event):
    '''
    Represents a layer in DNN
    '''
    def __init__(self, start_t:int, req_id:int, vvps:int, input_size:int) -> None:
        '''
        Parameters
        ----------
        start_t: see Event spec
        req_id: see Event spec
        vvps: number of VVPs in layer
        input_size: duration of each VVP (in ts)
        '''
        super().__init__(start_t, req_id)
        self.vvps:int = vvps
        self.input_size:int = input_size

    def gen_tasks(self) -> List[Task]:
        '''
        Returns
        -------
        tasks: list of Tasks for layer in DNN
        '''
        tasks = [Task(self.req_id, self.input_size) for _ in range(self.vvps)]
        return tasks


class Request(Event):
    '''
    Represents DNN with only fully-connected layers
    '''
    def __init__(self, start_t:int, layers:List[List[int]], req_id:int) -> None:
        '''
        Parameters
        ----------
        start_t: see Event spec
        layers: list of tuples that outline the input size and number of VVPs for each layer
        req_id: see Event spec
        '''
        super().__init__(start_t, req_id)
        self.layers = layers.copy() # to prevent aliasing

    def gen_job_dag(self, curr_time:int) -> Tuple[Job, List[List[int]]]:
        '''
        Parameters
        ----------
        curr_time: simulator's time (in ts)

        Returns
        -------
        job: Job corresponding to first layer of request
        dependent_layers: list of tuples that outline the input size and number of VVPs for each layer after this one (in same DNN)
        '''
        input_size, vvps = self.layers[0]
        job = Job(curr_time, self.req_id, vvps, input_size)
        dependent_layers = deepcopy(self.layers[1:]) # copied to prevent aliasing
        return job, dependent_layers
    

class LayerProgress():
    '''
    Represents state of layer for DNN
    '''
    def __init__(self, num_vvps_left:int, dependent_layers: List[List[int]]) -> None:
        '''
        Parameters
        ----------
        num_vvps_left: number of VVPs left to compute in current layer (non-negative integer)
        dependent_layers: list of tuples representing dimensions of subsequent layers, if exists
        '''
        self.num_vvps_left = num_vvps_left
        self.dependent_layers = deepcopy(dependent_layers) # copied to prevent aliasing