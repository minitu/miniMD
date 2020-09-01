#!/usr/bin/env python3
import os
import sys
import csv
import math
import statistics

# Number of GPUs
gpu_count = 6

if len(sys.argv) != 4:
  print('Please use', sys.argv[0], '[job name] [start node count] [end node count]')
  exit()

job_name = sys.argv[1]
start_node_count = int(sys.argv[2])
end_node_count = int(sys.argv[3])

csv_filename = job_name + '.csv'
csv_file = open(csv_filename, 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['Number of GPUs', 'MPI', 'error'])

node_count_list = []
cur_node_count = start_node_count
while cur_node_count <= end_node_count:
  node_count_list.append(cur_node_count)
  cur_node_count *= 2

for node_count in node_count_list:
  print('Node count:', str(node_count))
  comm_str = 'awk \'/MPI_proc/{getline; print}\' ' + job_name + '-n' + str(node_count) + '.* | cut -d " " -f 5'

  stream = os.popen(comm_str)
  lines = stream.readlines()
  timelist = list(map(lambda x: x * 10, list(map(float, list(map(str.rstrip, lines))))))
  print(timelist)

  avg = statistics.mean(timelist)
  stdev = statistics.stdev(timelist)

  writer.writerow([str(node_count * gpu_count) + "\\n(" + str(node_count) + ")", str(avg), str(stdev)])
