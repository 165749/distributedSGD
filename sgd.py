import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.optimizer import Optimizer


def build_distributed_model(model, lr, tracer, cuda=False, ignore_bn=False, no_overlap=False, all_reduce=False):
    class DistributedModel(model):
        def __init__(self, *args, **kwargs):
            super(DistributedModel, self).__init__(*args, **kwargs)
            if no_overlap:
                # If not overlapping communication and computation, skipping transmission during training
                for module in self.modules():
                    module.skip_layer = True
            if ignore_bn:
                bn_names = [name for name, module in self.named_modules() if isinstance(module, nn.BatchNorm2d)]
                for module in self.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.skip_layer = True
                self.parameters_with_names = [(name, para) for name, para in self.named_parameters()
                                              if name.rsplit('.', maxsplit=1)[0] not in bn_names]
            else:
                self.parameters_with_names = [(name, para) for name, para in self.named_parameters()]
            if all_reduce:
                # All workers in the communication group
                self.all_reduce_group = dist.new_group([i for i in range(1, dist.get_world_size(), 2)])
            self.worker_id = dist.get_rank()
            self.tracer = tracer
            self.parameters_buffer = []  # For receivers collecting model parameters
            self.senders = []
            self.receivers = []
            self.current_receiver = 0
            self.step_span = None
            self.span = None
            self.hooks = []
            self.register_hooks()
            self.no_overlap = no_overlap
            self.all_reduce = all_reduce
            for _, para in self.parameters_with_names:
                self.parameters_buffer.append(torch.zeros(para.data.size()))
            for name, module in self.named_modules():
                module.name = name

            # Send a empty gradients to fetch the initial model from the server
            # Send gradients to the server layer by layer
            with tracer.start_active_span('init'):
                # Wait for starting up
                with tracer.start_active_span('wait'):
                    dist.send(tensor=torch.zeros(1), dst=0)
                    dist.recv(tensor=torch.zeros(1), src=0)
                # If performing all-reduce, do not need to send parameters to the server
                if all_reduce:
                    return
                for name, para in reversed(self.parameters_with_names):
                    with tracer.start_active_span('send') as span:
                        span.set_tag('size', para.data.nelement() * para.data.element_size())
                        name = name.rsplit('.', maxsplit=1)
                        span.set_tag('layer', name[0])
                        span.set_tag('type', name[1])
                        dist.send(torch.zeros(para.data.size()), self.worker_id + 1)

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

        def send(self, tensor, layer, type):
            with self.tracer.start_active_span('send') as span:
                span.set_tag('size', tensor.nelement() * tensor.element_size())
                span.set_tag('layer', layer)
                span.set_tag('type', type)
                sender = dist.isend(tensor, self.worker_id + 1)
                self.senders.append(sender)

        def send_all_reduce(self, tensor, layer, type):
            with self.tracer.start_active_span('send') as span:
                span.set_tag('size', tensor.nelement() * tensor.element_size())
                span.set_tag('layer', layer)
                span.set_tag('type', type)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.all_reduce_group)

        def wait_all_senders(self):
            for i, sender in enumerate(self.senders):
                sender.wait()
            self.senders = []

        def reset_and_start_receivers(self):
            self.receivers = []
            self.current_receiver = 0
            for para in self.parameters_buffer:
                receiver = dist.irecv(para, self.worker_id + 1)
                self.receivers.append(receiver)

        def wait_receiver(self):
            with self.tracer.start_active_span('recv'):
                self.receivers[self.current_receiver].wait()
            with self.tracer.start_active_span('copy') as span:
                name, para = self.parameters_with_names[self.current_receiver]
                name = name.rsplit('.', maxsplit=1)
                span.set_tag('layer', name[0])
                span.set_tag('type', name[1])
                if cuda:
                    para.data.copy_(self.parameters_buffer[self.current_receiver])
                else:
                    para.data = self.parameters_buffer[self.current_receiver]
                self.current_receiver += 1

        def forward_pre_hook_fn(self, module, input):
            self.span.finish()
            if hasattr(module, 'skip_layer'):
                pass
            elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
                self.wait_receiver()
                if module.bias is not None:
                    self.wait_receiver()
            self.span = self.tracer.start_span('compute')
            self.span.set_tag('layer', module.name)

        def backward_hook_fn(self, module, input, output):
            self.span.set_tag('layer', module.name)
            self.span.finish()
            weight = None
            bias = None
            if hasattr(module, 'skip_layer'):
                pass
            elif isinstance(module, nn.Conv2d):
                weight = input[1]
                # Note: For GPU training, the argument input will only contain bias if bias=True for nn.Conv2d,
                # which is caused by a well-known issue of backward hooks in PyTorch. To bypass the issue, needs
                # to disable all the bias of nn.Conv2d in the model.
                if not cuda:
                    bias = input[2]
            elif isinstance(module, nn.BatchNorm2d):
                weight = input[1]
                bias = input[2]
            elif isinstance(module, nn.Linear):
                weight = input[2].t()
                bias = input[0]
            # Reverse order in the backward
            if bias is not None:
                with self.tracer.start_active_span('lr') as span:
                    span.set_tag('layer', module.name)
                    span.set_tag('type', 'bias')
                    grad = (-lr) * bias
                    grad = grad.cpu()
                    self.send(grad, module.name, 'bias')
            if weight is not None:
                with self.tracer.start_active_span('lr') as span:
                    span.set_tag('layer', module.name)
                    span.set_tag('type', 'weight')
                    grad = (-lr) * weight
                    grad = grad.cpu()
                    self.send(grad, module.name, 'weight')
            self.span = self.tracer.start_span('compute')

        def step_begin(self, epoch, i):
            # Inform the server starting next step (by setting tensor[0] to 0)
            tensor = torch.zeros(1)
            dist.send(tensor, self.worker_id + 1)

            if self.step_span is not None:
                self.step_span.finish()
            self.step_span = self.tracer.start_span('epoch {} step {}'.format(epoch, i))

            # If performing all-reduce, do not need to receive parameters from the server
            if self.all_reduce:
                return

            self.reset_and_start_receivers()
            if self.no_overlap:
                with self.tracer.start_active_span('downlink'):
                    for _ in self.parameters_with_names:
                        self.wait_receiver()

        def stop_training(self):
            # Inform the server about completion (by setting tensor[0] to inf)
            tensor = torch.zeros(1)
            tensor[0] = float('inf')
            dist.send(tensor, dist.get_rank() + 1)
            self.step_span.finish()

    return DistributedModel


class DownpourSGD(Optimizer):
    def __init__(self, params, lr, model):
        if lr is None or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,)
        self.model = model

        super(DownpourSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if self.model.no_overlap:
            # Learning rate
            lr = -self.param_groups[0]['lr']
            # Send gradients to the server layer by layer
            if self.model.all_reduce:
                with self.model.tracer.start_active_span('uplink'):
                    for name, para in self.model.parameters_with_names:
                        with self.model.tracer.start_active_span('lr') as span:
                            name = name.rsplit('.', maxsplit=1)
                            span.set_tag('layer', name[0])
                            span.set_tag('type', name[1])
                            self.model.send_all_reduce(para.grad, name[0], name[1])
                with self.model.tracer.start_active_span('update'):
                    for name, para in self.model.parameters_with_names:
                        para.data += lr * para.grad
            else:
                with self.model.tracer.start_active_span('uplink'):
                    for name, para in reversed(self.model.parameters_with_names):
                        with self.model.tracer.start_active_span('lr') as span:
                            name = name.rsplit('.', maxsplit=1)
                            span.set_tag('layer', name[0])
                            span.set_tag('type', name[1])
                            grad = lr * para.grad
                            grad = grad.cpu()
                            self.model.send(grad, name[0], name[1])
                    self.model.wait_all_senders()
        else:
            self.model.wait_all_senders()

        # Will pull parameters from the server, so no need to update internal parameters

        return loss
