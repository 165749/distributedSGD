import torch
import torch.optim
import torch.distributed as dist
from utils.trace import Tracer
from models.model import name_to_model


class ParameterServer:
    def __init__(self, args):
        self.worker_num = args.worker_num
        self.world_size = 2 * self.worker_num + 1  # size of communication group
        print("Creating Parameter Server with {} worker...".format(self.worker_num))

        model = name_to_model[args.model](num_classes=10)
        # Store model in the shared memory
        model = model.cuda()
        model.share_memory()
        if args.ignore_bn:
            bn_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d)]
            parameters_with_names = [(name, para) for name, para in model.named_parameters() if
                                     name.rsplit('.', maxsplit=1)[0] not in bn_names]
        else:
            parameters_with_names = [(name, para) for name, para in model.named_parameters()]
        self.parameters_with_names = parameters_with_names
        self.sync = args.sync

    def run(self):
        threads = []
        torch.multiprocessing.set_start_method('spawn')
        barrier = torch.multiprocessing.Barrier(self.worker_num)
        for server_id in range(2, 2 * self.worker_num + 1, 2):
            thread = torch.multiprocessing.Process(target=ParameterServer.receive,
                                                   args=(self.parameters_with_names, server_id, self.worker_num,
                                                         barrier, self.sync))
            thread.start()
            threads.append(thread)

        # Initialize communication group
        print("Initializing server with rank {}".format(0))
        dist.init_process_group('gloo', rank=0, world_size=2 * self.worker_num + 1)
        print("server 0 initialized")

        for thread in threads:
            thread.join()

        dist.destroy_process_group()
        print("server 0 finished")

    @classmethod
    def receive(cls, parameters_with_names, server_id, worker_num, barrier, sync):
        torch.autograd.set_grad_enabled(False)
        # Set up communication group
        print("Initializing server with rank {}".format(server_id))
        dist.init_process_group('gloo', rank=server_id, world_size=2 * worker_num + 1)
        print("server {} initialized".format(server_id))

        # Start tracer for each server
        tracer = Tracer(cuda=False)

        global_model = [para.data for _, para in parameters_with_names]
        layer_name = [name.rsplit('.', maxsplit=1) for name, _ in parameters_with_names]

        with tracer.start_active_span('server {}'.format(server_id)):
            span_step = tracer.start_span("init")
            gradient_buffers = [torch.zeros(para.data.size()) for _, para in parameters_with_names]
            step_num = 0

            # Wait for starting up
            with tracer.start_active_span('wait'):
                barrier.wait()
                dist.recv(tensor=torch.zeros(1), src=server_id - 1)
                dist.send(tensor=torch.zeros(1), dst=server_id - 1)

            while True:
                receivers = []
                # Receive gradients in reverse order
                for buffer in reversed(gradient_buffers):
                    receivers.append(dist.irecv(tensor=buffer, src=server_id - 1))
                for i, receiver in enumerate(receivers):
                    with tracer.start_active_span('recv') as span:
                        span.set_tag('layer', layer_name[-1 - i][0])
                        span.set_tag('type', layer_name[-1 - i][1])
                        receiver.wait()
                        if not sync:  # For async
                            with tracer.start_active_span('update'):
                                global_model[-1 - i].add_(gradient_buffers[-1 - i].cuda())
                                gradient_buffers[-1 - i].copy_(global_model[-1 - i])
                if sync:  # For sync
                    with tracer.start_active_span('update'):
                        for i, gradients in enumerate(gradient_buffers):
                            global_model[i].add_(gradients.cuda())
                    with tracer.start_active_span('collect'):
                        for i, gradients in enumerate(gradient_buffers):
                            gradients.copy_(global_model[i])
                    with tracer.start_active_span('barrier'):
                        barrier.wait()
                # TODO: cuda synchronization
                torch.cuda.synchronize()

                tensor = torch.zeros(1)
                dist.recv(tensor=tensor, src=server_id - 1)
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
                        dist.send(para, dst=server_id - 1)
        dist.destroy_process_group()
        tracer.export_traces("server{}.json".format(server_id))
        print("server {} finished".format(server_id))
