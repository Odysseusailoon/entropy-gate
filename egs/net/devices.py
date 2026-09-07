"""Read-only admission check for shared GPUs. Never evicts an existing job."""
import argparse
import csv
import io
import subprocess


def inventory():
    def query(arguments):
        output = subprocess.check_output(['nvidia-smi', *arguments, '--format=csv,noheader,nounits'], text=True)
        return list(csv.reader(io.StringIO(output), skipinitialspace=True))
    cards = query(['--query-gpu=index,uuid,name,memory.used,utilization.gpu'])
    jobs = query(['--query-compute-apps=gpu_uuid,pid'])
    return cards, jobs


def validate_devices(cards, jobs, devices, required_name='H200'):
    if not devices or len(set(devices)) != len(devices):
        raise RuntimeError('select distinct GPU indices')
    by_id = {r[0]: r for r in cards}
    occupied = {r[0] for r in jobs if r}
    selected = []
    for device in devices:
        if device not in by_id:
            raise RuntimeError(f'GPU {device} is unavailable')
        index, uuid, name, memory, utilization = by_id[device]
        if required_name not in name:
            raise RuntimeError(f'GPU {index} is {name}; this launch requires {required_name}')
        if uuid in occupied or float(memory) > 256 or float(utilization) > 5:
            raise RuntimeError(f'GPU {index} is occupied; leave its processes untouched and select idle devices')
        selected.append({'index': index, 'uuid': uuid, 'name': name})
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', required=True, help='comma-separated physical GPU indices')
    parser.add_argument('--required-name', default='H200')
    args = parser.parse_args()
    for row in validate_devices(*inventory(), args.devices.split(','), args.required_name):
        print(f"GPU {row['index']}: {row['name']}, currently idle ({row['uuid']})")


if __name__ == '__main__':
    main()
