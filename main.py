import os
import argparse
import torch.distributed as dist

from server import ParameterServer
from worker import Worker

from models.model import name_to_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='distributed SGD')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA for training')
    parser.add_argument('--evaluate-per-epoch', action='store_true', default=False,
                        help='whether to evaluate after each epoch')
    parser.add_argument('--log-interval', type=int, default=-1, metavar='N', help='how often to evaluate and print out')
    parser.add_argument('--display-time', action='store_true', default=False,
                        help='whether to displace time of each training step')
    parser.add_argument('--ignore-bn', action='store_true', default=False,
                        help='whether to ignore bn layers when transmitting parameters')
    parser.add_argument('--no-overlap', action='store_true', default=False,
                        help='whether not to overlap communication and computation')
    parser.add_argument('--all-reduce', action='store_true', default=False,
                        help='whether to use all_reduce collective communications (only for CPU)')
    parser.add_argument('--worker-id', type=int, default=0, metavar='N', help='rank of the current worker')
    parser.add_argument('--worker-num', type=int, default=1, metavar='N', help='number of workers in the training')
    parser.add_argument('--gpu-per-worker', type=int, default=1, metavar='N', help='number of gpus per worker')
    parser.add_argument('--server', action='store_true', default=False, help='server node?')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='which dataset to train on')
    parser.add_argument('--model', type=str, default='alexnet', help='which model to train')
    parser.add_argument('--num-batches', type=int, default=100, metavar='N',
                        help='number of batches to train (default: 100)')
    parser.add_argument('--image-size', type=int, default=-1, metavar='N',
                        help='size of images to train (default: -1, which will set the recommended image size for some models)')
    parser.add_argument('--master', type=str, default='localhost', help='ip address of the master (server) node')
    parser.add_argument('--port', type=str, default='2222', help='port on master node to communicate with')
    parser.add_argument('--interface', type=str, default='none', help='Choose network interface to use')
    parser.add_argument('--threads', type=int, default='0', help='How many threads to run')
    parser.add_argument('--sync', action='store_true', default=False, help='Enable synchronous training')
    args = parser.parse_args()
    print(args)

    os.environ['MASTER_ADDR'] = args.master
    os.environ['MASTER_PORT'] = args.port
    os.environ['NCCL_IB_DISABLE'] = '1'  # TODO: enable for infiniband
    if args.interface != "none":
        os.environ['GLOO_SOCKET_IFNAME'] = args.interface
        os.environ['GLOO_SOCKET_IFNAME'] = args.interface
        print('Set network interface {}'.format(args.interface))

    if args.model not in name_to_model.keys():
        raise Exception("Not implemented yet: {}".format(args.model))

    if args.server:
        server = ParameterServer(args=args)
        server.run()
    else:
        worker = Worker(args=args)
        if args.gpu_per_worker > 1:
            assert args.no_overlap is True
            assert args.all_reduce is False  # TODO: implementation
            worker.run_multi_gpu(args)
        else:
            worker.run(args)
    dist.destroy_process_group()
