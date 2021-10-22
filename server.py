import torch
import torch.optim
import torch.distributed as dist
from utils.trace import Tracer
from models.model import name_to_model


class ParameterServer:
    def __init__(self, args):
        self.worker_num = args.worker_num
        self.gpu_per_worker = args.gpu_per_worker
        self.world_size = (1 + self.gpu_per_worker) * self.worker_num + 1  # size of communication group
        print("Creating Parameter Server with {} worker...".format(self.worker_num))

        model = name_to_model[args.model](num_classes=10)
        # Store model in the shared memory
        if args.cuda:
            model = model.cuda()
        model.share_memory()
        if args.ignore_bn:
            bn_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d)]
            parameters_with_names = [(name, para) for name, para in model.named_parameters() if
                                     name.rsplit('.', maxsplit=1)[0] not in bn_names]
        else:
            parameters_with_names = [(name, para) for name, para in model.named_parameters()]
        self.parameters_with_names = parameters_with_names
        self.cuda = args.cuda
        self.sync = args.sync
        self.all_reduce = args.all_reduce

    def run(self):
        threads = []
        torch.multiprocessing.set_start_method('spawn')
        barrier = torch.multiprocessing.Barrier(self.worker_num) if self.sync else None
        for server_id in range(1 + self.gpu_per_worker, self.world_size, 1 + self.gpu_per_worker):
            thread = torch.multiprocessing.Process(target=ParameterServer.receive,
                                                   args=(self.parameters_with_names, server_id, self.worker_num,
                                                         self.gpu_per_worker, self.cuda, barrier, self.all_reduce))
            thread.start()
            threads.append(thread)

        # Initialize communication group
        print("Initializing server with rank {}".format(0))
        dist.init_process_group('nccl', rank=0, world_size=self.world_size)
        if self.all_reduce:
            # Create new group for all workers to perform all_reduce
            dist.new_group([i for i in range(1, dist.get_world_size(), 2)])
        if self.gpu_per_worker > 1:
            for worker_id in range(self.worker_num):
                dist.new_group([(1 + self.gpu_per_worker) * worker_id + k + 1 for k in range(self.gpu_per_worker)])
        print("server 0 initialized")

        for thread in threads:
            thread.join()

        dist.destroy_process_group()
        print("server 0 finished")

    @classmethod
    def receive(cls, parameters_with_names, server_id, worker_num, gpu_per_worker, cuda, barrier, all_reduce):
        if cuda:
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
        else:
            device = torch.device('cpu')

        torch.autograd.set_grad_enabled(False)
        # Set up communication group
        print("Initializing server with rank {}".format(server_id))
        dist.init_process_group('nccl', rank=server_id, world_size=(1 + gpu_per_worker) * worker_num + 1)
        if all_reduce:
            # Create new group for all workers to perform all_reduce
            dist.new_group([i for i in range(1, dist.get_world_size(), 2)])
        if gpu_per_worker > 1:
            for worker_id in range(worker_num):
                dist.new_group([(1 + gpu_per_worker) * worker_id + k + 1 for k in range(gpu_per_worker)])
        print("server {} initialized".format(dist.get_rank()))

        # Start tracer for each server
        tracer = Tracer(cuda=cuda)

        global_model = [para.data for _, para in parameters_with_names]
        layer_name = [name.rsplit('.', maxsplit=1) for name, _ in parameters_with_names]

        with tracer.start_active_span('server {}'.format(server_id)):
            span_step = tracer.start_span("init")
            gradient_buffers = [torch.zeros(para.data.size(), device=device) for _, para in parameters_with_names]
            step_num = 0

            # Wait for starting up
            with tracer.start_active_span('wait'):
                dist.recv(tensor=torch.zeros(1, device=device), src=server_id - gpu_per_worker)
                dist.send(tensor=torch.zeros(1, device=device), dst=server_id - gpu_per_worker)
                barrier.wait()

            while True:
                if all_reduce:
                    tensor = torch.zeros(1, device=device)
                    dist.recv(tensor=tensor, src=server_id - gpu_per_worker)
                    span_step.finish()
                    if tensor[0] == float('inf'):
                        break
                    span_step = tracer.start_span('step {}'.format(step_num))
                    step_num += 1
                else:
                    receivers = []
                    # Receive gradients in the reverse order
                    for buffer in reversed(gradient_buffers):
                        receivers.append(dist.irecv(tensor=buffer, src=server_id - gpu_per_worker))
                    for i, receiver in enumerate(receivers):
                        with tracer.start_active_span('recv') as span:
                            span.set_tag('layer', layer_name[-1 - i][0])
                            span.set_tag('type', layer_name[-1 - i][1])
                            receiver.wait()
                            if barrier is None:  # For async
                                with tracer.start_active_span('update'):
                                    global_model[-1 - i].add_(gradient_buffers[-1 - i])
                                    gradient_buffers[-1 - i].copy_(global_model[-1 - i])
                    if barrier is not None:  # For sync
                        with tracer.start_active_span('update'):
                            for i, gradients in enumerate(gradient_buffers):
                                global_model[i].add_(gradients)
                        with tracer.start_active_span('collect'):
                            for i, gradients in enumerate(gradient_buffers):
                                gradients.copy_(global_model[i])
                    tensor = torch.zeros(1, device=device)
                    dist.recv(tensor=tensor, src=server_id - gpu_per_worker)
                    if barrier is not None:
                        with tracer.start_active_span('barrier'):
                            barrier.wait()
                    span_step.finish()
                    if tensor[0] == float('inf'):
                        break
                    span_step = tracer.start_span('step {}'.format(step_num))
                    step_num += 1
                    for i, para in enumerate(gradient_buffers):
                        with tracer.start_active_span('send') as span:
                            span.set_tag('size', para.nelement() * para.element_size())
                            span.set_tag('layer', layer_name[i][0])
                            span.set_tag('type', layer_name[i][1])
                            dist.send(para, dst=server_id - gpu_per_worker)
        dist.destroy_process_group()
        tracer.export_traces("server{}.json".format(server_id))
        print("server {} finished".format(server_id))
