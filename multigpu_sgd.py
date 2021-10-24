import os
import torch
import torch.nn as nn
import torch.distributed as dist
from utils.trace import Tracer


def build_multi_gpu_model(model, worker_id, worker_num, gpu_id, gpu_per_worker, tracer, ignore_bn=False):
    class MultiGPUModel(model):
        def __init__(self, *args, **kwargs):
            # Support since PyTorch 1.10
            dist._DEFAULT_FIRST_BUCKET_BYTES = 1

            master_addr = os.environ['MASTER_ADDR']
            master_port = os.environ['MASTER_PORT']
            os.environ['MASTER_ADDR'] = "localhost"
            os.environ['MASTER_PORT'] = "2223"
            dist.init_process_group('nccl', rank=gpu_id, world_size=gpu_per_worker)
            print("Worker {} (id: {}, gpu: {}) initialized".format(dist.get_rank(), worker_id, gpu_id))

            super(MultiGPUModel, self).__init__(*args, **kwargs)

            if ignore_bn:
                bn_names = [name for name, module in self.named_modules() if isinstance(module, nn.BatchNorm2d)]
                for module in self.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.skip_layer = True
                self.parameters_with_names = [(name.replace('module.', ''), para) for name, para in self.named_parameters()
                                              if name.rsplit('.', maxsplit=1)[0] not in bn_names]
            else:
                self.parameters_with_names = [(name.replace('module.', ''), para) for name, para in self.named_parameters()]
            self.gpu_id = gpu_id
            self.gpu_per_worker = gpu_per_worker
            if self.gpu_id == 0:
                # Only GPU 0 performs downlink and uplink
                self.server_rank = dist.get_rank() + self.gpu_per_worker
            self.tracer = tracer
            self.parameters_buffer = []  # For receivers collecting model parameters
            self.step_span = None
            self.span = None
            self.hooks = []
            self.register_hooks()

            for name, module in self.named_modules():
                module.name = name.replace('module.', '')
            for _, para in self.parameters_with_names:
                self.parameters_buffer.append(torch.zeros(para.data.size()).share_memory_())

            if gpu_id == 0:
                names = [name for name, _ in self.parameters_with_names]
                self.q1 = torch.multiprocessing.Queue()
                self.q2 = torch.multiprocessing.Queue()
                self.handler_thread = torch.multiprocessing.spawn(
                    update_handler, nprocs=1, join=False,
                    args=(self.parameters_buffer, names, worker_id, worker_num, self.q1, self.q2, master_addr, master_port))

        def register_hooks(self):
            print('Register hooks!')
            for layer in self.modules():
                hook = layer.register_forward_pre_hook(self.forward_pre_hook_fn)
                self.hooks.append(hook)
                hook = layer.register_backward_hook(self.backward_hook_fn)
                self.hooks.append(hook)

        def remove_hooks(self):
            print('Remove hooks!')
            for hook in self.hooks:
                hook.remove()

        def init_tracer_span(self):
            self.span = self.tracer.start_span('compute')

        def finish_tracer_span(self):
            self.span.finish()

        def forward_pre_hook_fn(self, module, input):
            pass
            self.span.finish()
            self.span = self.tracer.start_span('compute')
            self.span.set_tag('layer', module.name)

        def backward_hook_fn(self, module, input, output):
            pass
            self.span.set_tag('layer', module.name)
            self.span.finish()
            self.span = self.tracer.start_span('compute')

        def init(self):
            # Send a empty gradients to fetch the initial model from the server
            # Send gradients to the server layer by layer
            with tracer.start_active_span('init'):
                if self.gpu_id == 0:
                    with tracer.start_active_span('wait'):
                        self.q1.put('sync')
                        self.q2.get()
                        self.q1.put('send')
                        self.q2.get()

        def step_begin(self, step_idx):
            if self.gpu_id == 0:
                self.q1.put('update')
                self.q2.get()
            # Synchronize all GPUs
            dist.all_reduce(torch.zeros(1).cuda(), op=dist.ReduceOp.SUM)

            if self.step_span is not None:
                self.step_span.finish()
            self.step_span = self.tracer.start_span('step {}'.format(step_idx))

            if self.gpu_id == 0:
                with self.tracer.start_active_span('downlink'):
                    self.q1.put('recv')
                    self.q2.get()

                    for layer_idx, buffer in enumerate(self.parameters_buffer):
                        with self.tracer.start_active_span('copy') as span:
                            name, para = self.parameters_with_names[layer_idx]
                            name = name.rsplit('.', maxsplit=1)
                            span.set_tag('layer', name[0])
                            span.set_tag('type', name[1])
                            para.data.copy_(buffer)
            # Synchronize all GPUs
            dist.all_reduce(torch.zeros(1).cuda(), op=dist.ReduceOp.SUM)

        def step_finish(self, lr):
            with self.tracer.start_active_span('uplink'):
                if self.gpu_id == 0:
                    layer_idx = len(self.parameters_buffer) - 1
                    for buffer in reversed(self.parameters_buffer):
                        with self.tracer.start_active_span('lr') as span:
                            name, para = self.parameters_with_names[layer_idx]
                            name = name.rsplit('.', maxsplit=1)
                            span.set_tag('layer', name[0])
                            span.set_tag('type', name[1])
                            buffer.copy_(lr * para.grad)
                            layer_idx -= 1

                    self.q1.put('send')
                    self.q2.get()

        def stop_training(self):
            if gpu_id == 0:
                self.q1.put('stop')
                self.q2.get()
                self.handler_thread.join()
            # Synchronize all GPUs
            dist.all_reduce(torch.zeros(1).cuda(), op=dist.ReduceOp.SUM)
            self.step_span.finish()

        def _allreduce_fut(
                self, process_group: dist.ProcessGroup, tensor: torch.Tensor
        ) -> torch.futures.Future[torch.Tensor]:
            span = self.tracer.start_span('allreduce')
            span.set_tag('size', tensor.nelement() * tensor.element_size())
            group_to_use = process_group if process_group is not None else dist.group.WORLD

            # Apply the division first to avoid overflow, especially for FP16.
            tensor.div_(group_to_use.size())

            def complete(fut):
                span.finish()
                return fut.value()[0]

            return (
                dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future().then(complete)
            )

        def communication_fn(self, process_group: dist.ProcessGroup, bucket: dist.GradBucket
                             ) -> torch.futures.Future[torch.Tensor]:
            return self._allreduce_fut(process_group, bucket.buffer())

    return MultiGPUModel


def update_handler(idx, parameters_buffer, names, worker_id, worker_num, q1, q2, master_addr, master_port):
    tracer = Tracer(cuda=False)

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group('gloo', rank=2 * worker_id + 1, world_size=2 * worker_num + 1)
    print("Worker {} cpu initialized".format(worker_id))
    server_rank = dist.get_rank() + 1

    root_span = tracer.start_span('worker {}'.format(dist.get_rank()))
    step_idx = -1
    step_span = tracer.start_span('init')

    while True:
        instruction = q1.get()
        if instruction == "sync":
            with tracer.start_active_span('wait'):
                # Wait for starting up
                dist.send(tensor=torch.zeros(1), dst=server_rank)
                dist.recv(tensor=torch.zeros(1), src=server_rank)
        elif instruction == "update":
            # Wait for server to finish updates
            dist.send(torch.zeros(1), dst=server_rank)
            step_span.finish()
            step_idx += 1
            step_span = tracer.start_span('step {}'.format(step_idx))
        elif instruction == "send":
            with tracer.start_active_span('uplink'):
                layer_idx = len(parameters_buffer) - 1
                for buffer in reversed(parameters_buffer):
                    with tracer.start_active_span('send') as span:
                        span.set_tag('size', buffer.nelement() * buffer.element_size())
                        name = names[layer_idx].rsplit('.', maxsplit=1)
                        span.set_tag('layer', name[0])
                        span.set_tag('type', name[1])
                        dist.send(buffer, dst=server_rank)
                        layer_idx -= 1
        elif instruction == "recv":
            with tracer.start_active_span('downlink'):
                receivers = []
                # Receive gradients in reverse order
                for buffer in parameters_buffer:
                    receivers.append(dist.irecv(tensor=buffer, src=server_rank))
                for layer_idx, receiver in enumerate(receivers):
                    with tracer.start_active_span('recv') as span:
                        name = names[layer_idx].rsplit('.', maxsplit=1)
                        span.set_tag('layer', name[0])
                        span.set_tag('type', name[1])
                        receiver.wait()
        elif instruction == "stop":
            # Inform the server about completion (by setting tensor[0] to inf)
            tensor = torch.zeros(1)
            tensor[0] = float('inf')
            dist.send(tensor, dst=server_rank)
            q2.put(0)
            step_span.finish()
            break
        q2.put(0)

    root_span.finish()

    tracer.export_traces("worker{}.json".format(dist.get_rank()))
    dist.destroy_process_group()
