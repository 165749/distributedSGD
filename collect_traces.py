import gzip
import json
import sys

if __name__ == '__main__':
    assert len(sys.argv) > 2
    # Number of machines
    worker_num = int(sys.argv[1])
    # GPUs per worker
    gpu_per_worker = int(sys.argv[2])
    # Output file
    file_name = sys.argv[3]
    # Decentralized settings
    all_reduce = True if len(sys.argv) > 4 and "--all-reduce" in sys.argv[4] else False

    if ".gz" not in file_name:
        file_name += ".gz"
    with gzip.open(file_name, "wt") as output_file:
        output_dict = {}
        if all_reduce:
            # For all_reduce case, only consider workers
            for worker_id in range(worker_num):
                with gzip.open(f"worker{worker_id}.json.gz") as f:
                    output_dict[f"worker{worker_id}"] = json.load(f)
        else:
            # Otherwise consider both workers and servers
            for worker_id in range(1, 2 * worker_num + 1, 2):
                with gzip.open(f"worker{worker_id}.json.gz") as f:
                    output_dict[f"worker{worker_id}"] = json.load(f)
                for gpu_id in range(gpu_per_worker):
                    with gzip.open(f"worker{worker_id}_gpu{gpu_id}.json.gz") as f:
                        output_dict[f"worker{worker_id}_gpu{gpu_id}"] = json.load(f)
            for server_id in range(2, 2 * worker_num + 1, 2):
                with gzip.open(f"server{server_id}.json.gz") as f:
                    output_dict[f"server{server_id}"] = json.load(f)
        json.dump(output_dict, output_file)
