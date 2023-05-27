class Event():
    def __init__(self, start_t, req_id):
        '''
        Parameters
        ----------
        start_t: time (in ts of simulator) of event's arrival (non-negative integer)
        req_id: identifier of request that the event is associated with (a natural number)
        '''
        self.start_t = start_t
        self.req_id = req_id


class Request(Event):
    '''
    Represents DNN with only fully-connected layers
    '''
    def __init__(self, start_t, layers, req_id):
        '''
        Parameters
        ----------
        start_t: see Event spec
        layers: list of tuples that outline the input size and number of VVPs for each layer
        req_id: see Event spec
        '''
        super().__init__(start_t, req_id)
        self.layers = layers.copy() # to prevent aliasing

    def gen_job_dag(self, curr_time):
        '''
        Parameters
        ----------
        curr_time: simulator's time (in ts)

        Returns
        -------
        job: Jobs corresponding to first layer of request
        dependent_layers: list of tuples that outline the input size and number of VVPs for each layer after this one
        '''
        input_size, vvps = self.layers[0]
        job = Job(curr_time, self.req_id, vvps, input_size)
        dependent_layers = self.layers[1:].copy() # to prevent aliasing
        return job, dependent_layers


class Job(Event):
    '''
    Represents a layer in DNN
    '''
    def __init__(self, start_t, req_id, vvps, input_size):
        '''
        Parameters
        ----------
        start_t: see Event spec
        req_id: see Event spec
        vvps: number of VVPs in layer
        input_size: duration of each VVP (in ts)
        '''
        super().__init__(start_t, req_id)
        self.vvps = vvps
        self.input_size = input_size

    def gen_tasks(self):
        '''
        Returns
        -------
        tasks: list of Tasks for layer in DNN
        '''
        tasks = [Task(self.req_id, self.input_size) for _ in range(self.vvps)]
        return tasks


class Task():
    def __init__(self, req_id, size):
        self.req_id = req_id
        self.size = size