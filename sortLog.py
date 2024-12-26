import re
from datetime import datetime

log_file = '/Users/bernd/python/XOR/mylog.txt'
sorted_log_file = '/Users/bernd/python/XOR/mylog_sorted.txt'

with open(log_file, 'r') as f:
    lines = f.readlines()

blocks = {}
current_pid = None
block_start_time = {}

timestamp_regex = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[PID (\d+)\]')

for line in lines:
    match = timestamp_regex.match(line)
    if match:
        timestamp_str, pid = match.groups()
        if pid not in blocks:
            blocks[pid] = []
            block_start_time[pid] = datetime.strptime(
                timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
        blocks[pid].append(line)
    else:
        if current_pid:
            blocks[current_pid].append(line)

sorted_pids = sorted(block_start_time, key=block_start_time.get)

with open(sorted_log_file, 'w') as f:
    for pid in sorted_pids:
        for line in blocks[pid]:
            f.write(line)
