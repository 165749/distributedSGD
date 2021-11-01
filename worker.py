import os
import time
import torchvision
import torchvision.transforms as transforms
import torch
import torch.distributed as dist
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from models.model import name_to_model
from sgd import DownpourSGD
from sgd import build_distributed_model, build_allreduce_model
from multigpu_sgd import build_multi_gpu_model
from utils.trace import Tracer


class Worker:
    def __init__(self, args):
        # Set number of threads for each worker
        if args.threads > 0:
            torch.set_num_threads(args.threads)
        print('number of threads: {}'.format(torch.get_num_threads()))

        # Prepare data before joining communication group
        self.train_loader, self.test_loader = Worker.prepare_data(args)
        self.logs = []

    def run_multi_gpu(self, args):
        del self.train_loader
        del self.test_loader
        print("Creating Worker {} with {} GPUs...".format(args.worker_id, args.gpu_per_worker))
        torch.multiprocessing.spawn(Worker.train_multi_gpu, nprocs=args.gpu_per_worker, args=(args,))

    @classmethod
    def train_multi_gpu(cls, gpu_id, args):
        torch.cuda.set_device(gpu_id)

        train_loader, _ = cls.prepare_data(args)

        tracer = Tracer(cuda=True)

        model = name_to_model[args.model](num_classes=10)
        model = model.to(gpu_id)
        model = build_multi_gpu_model(
            torch.nn.parallel.DistributedDataParallel, worker_id=args.worker_id, worker_num=args.worker_num,
            gpu_id=gpu_id, gpu_per_worker=args.gpu_per_worker, tracer=tracer, ignore_bn=args.ignore_bn,
            allreduce=args.all_reduce
        )(model, device_ids=[gpu_id], bucket_cap_mb=500)
        if args.all_reduce:
            model.register_comm_hook(model.local_group, model.communication_fn)  # Since PyTorch 1.8
        else:
            model.register_comm_hook(None, model.communication_fn)  # Since PyTorch 1.8

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, min_lr=1e-3)

        root_span = tracer.start_span('worker {}'.format(dist.get_rank()))

        # train
        model.train()
        model.init()

        steps_per_epoch = len(train_loader)
        for epoch in range(args.epochs):  # loop over the dataset multiple times
            print("Training for epoch {}".format(epoch))
            for i, data in enumerate(train_loader):
                print('step {}'.format(i))

                if args.display_time:
                    start = time.time()

                # Inform server starting next step (i.e., starting pushing the model to the worker)
                model.step_begin(epoch * steps_per_epoch + i)

                with tracer.start_active_span('prepare_data'):
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()

                with tracer.start_active_span('zero_grad'):
                    # Clear the parameter gradients
                    optimizer.zero_grad()

                with tracer.start_active_span('forward'):
                    model.init_tracer_span()
                    outputs = model(inputs)
                    model.finish_tracer_span()

                with tracer.start_active_span('loss'):
                    loss = torch.nn.functional.cross_entropy(outputs, labels)

                with tracer.start_active_span('backward'):
                    model.init_tracer_span()
                    loss.backward()
                    model.finish_tracer_span()

                # optimizer.step()
                model.step_finish(args.lr)

                if args.display_time:
                    end = time.time()
                    print('time: {}'.format(end - start))

                if args.num_batches <= epoch * steps_per_epoch + i + 1:
                    break

            if args.num_batches <= (epoch + 1) * steps_per_epoch:
                break

        # Stop training
        model.stop_training()
        root_span.finish()
        torch.cuda.synchronize(device=gpu_id)
        if args.all_reduce:
            tracer.export_traces("worker{}_gpu{}.json".format(args.worker_id, gpu_id))
        else:
            tracer.export_traces("worker{}_gpu{}.json".format(2 * args.worker_id + 1, gpu_id))

        dist.destroy_process_group()
        print('Finished Training')

    def run(self, args):
        if args.cuda and args.all_reduce:
            dist.init_process_group('nccl', rank=args.worker_id, world_size=args.worker_num)
            print("worker {} initialized".format(dist.get_rank()))
            # Support since PyTorch 1.10
            dist._DEFAULT_FIRST_BUCKET_BYTES = 1

            tracer = Tracer(cuda=args.cuda)
            model = name_to_model[args.model](num_classes=10)
            model = model.cuda()
            model = build_allreduce_model(
                torch.nn.parallel.DistributedDataParallel, tracer=tracer)(model, bucket_cap_mb=500)
            model.register_comm_hook(None, model.communication_fn)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)
        else:
            dist.init_process_group('gloo', rank=2 * args.worker_id + 1, world_size=2 * args.worker_num + 1)
            print("worker {} initialized".format(dist.get_rank()))

            tracer = Tracer(cuda=args.cuda)
            model = build_distributed_model(name_to_model[args.model], lr=args.lr, tracer=tracer, cuda=args.cuda,
                                            ignore_bn=args.ignore_bn, no_overlap=args.no_overlap,
                                            all_reduce=args.all_reduce)(num_classes=10)

            optimizer = DownpourSGD(model.parameters(), lr=args.lr, model=model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, min_lr=1e-3)

        root_span = tracer.start_span('worker {}'.format(dist.get_rank()))

        # train
        model.train()
        if args.cuda:
            model = model.cuda()
        model.init()

        steps_per_epoch = len(self.train_loader)
        for epoch in range(args.epochs):  # loop over the dataset multiple times
            print("Training for epoch {}".format(epoch))
            for i, data in enumerate(self.train_loader):
                print('step {}'.format(i))

                if args.display_time:
                    start = time.time()

                # Inform server starting next step (i.e., starting pushing the model to the worker)
                model.step_begin(epoch * steps_per_epoch + i)

                with tracer.start_active_span('prepare_data'):
                    inputs, labels = data
                    if args.cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()

                with tracer.start_active_span('zero_grad'):
                    # Clear the parameter gradients
                    optimizer.zero_grad()

                with tracer.start_active_span('forward'):
                    model.init_tracer_span()
                    outputs = model(inputs)
                    model.finish_tracer_span()

                with tracer.start_active_span('loss'):
                    loss = torch.nn.functional.cross_entropy(outputs, labels)

                with tracer.start_active_span('backward'):
                    model.init_tracer_span()
                    loss.backward()
                    model.finish_tracer_span()
                optimizer.step()

                if args.display_time:
                    end = time.time()
                    print('time: {}'.format(end - start))

                if args.num_batches <= epoch * steps_per_epoch + i + 1:
                    break
                if args.log_interval > 0 and i % args.log_interval == 0 and i > 0:
                    self.store_state(model, i, outputs, labels, loss, args)

            if args.num_batches <= (epoch + 1) * steps_per_epoch:
                break

            if args.evaluate_per_epoch:
                val_loss, val_accuracy = self.evaluate(model, args, verbose=True)
                scheduler.step(val_loss)

        # Stop training
        model.stop_training()
        root_span.finish()
        tracer.export_traces("worker{}.json".format(dist.get_rank()))

        self.export_log(args)
        print('Finished Training')

    @classmethod
    def prepare_data(cls, args):
        image_size = 0
        if args.image_size < 0:
            _name_to_image_size = {
                "alexnet": 224 + 3,
                "mobilenet": 224,
                "googlenet": 224,
                "inception3": 299,
                "resnet50": 224,
                "resnet101": 224,
                "resnet152": 224,
                "vgg11": 224,
                "vgg13": 224,
                "vgg16": 224,
                "vgg19": 256,
            }
            if args.model in _name_to_image_size.keys():
                image_size = _name_to_image_size[args.model]
        else:
            image_size = args.image_size

        assert image_size >= 0
        if image_size == 0:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if args.dataset == 'MNIST':
            train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        else:
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False,
                                                  num_workers=1)
        return train_loader, test_loader

    def store_state(self, model, index, outputs, labels, loss, args):
        _, predicted = torch.max(outputs, 1)
        if args.cuda:
            labels = labels.view(-1).cpu().numpy()
            predicted = predicted.view(-1).cpu().numpy()
        accuracy = accuracy_score(predicted, labels)
        test_loss, test_accuracy = self.evaluate(model, args)
        log_obj = {
            'timestamp': time.time(),
            'iteration': index,
            'training_loss': loss.item(),
            'training_accuracy': accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
        print("Timestamp: {timestamp} | "
              "Iteration: {iteration:6} | "
              "Loss: {training_loss:6.4f} | "
              "Accuracy : {training_accuracy:6.4f} | "
              "Test Loss: {test_loss:6.4f} | "
              "Test Accuracy: {test_accuracy:6.4f}".format(**log_obj))
        self.logs.append(log_obj)

    def evaluate(self, model, args, verbose=False):
        if args.dataset == 'MNIST':
            classes = [str(i) for i in range(10)]
        else:
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        model.eval()

        # Temporarily remove hooks for evaluation
        model.remove_hooks()

        test_loss = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data

                if args.cuda:
                    images, labels = images.cuda(), labels.cuda()

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_loss += torch.nn.functional.cross_entropy(outputs, labels).item()

        if args.cuda:
            labels = labels.view(-1).cpu().numpy()
            predicted = predicted.view(-1).cpu().numpy()

        test_accuracy = accuracy_score(predicted, labels)
        if verbose:
            print('Loss: {:.3f}'.format(test_loss))
            print('Accuracy: {:.3f}'.format(test_accuracy))
            print(classification_report(predicted, labels, target_names=classes))

        # Restore hooks
        model.register_hooks()

        return test_loss, test_accuracy

    def export_log(self, args):
        df = pd.DataFrame(self.logs)
        print(df)
        if not os.path.exists('log'):
            os.mkdir('log')
        if args.cuda:
            df.to_csv('log/gpu.csv', index_label='index')
        else:
            df.to_csv('log/cpu.csv', index_label='index')
