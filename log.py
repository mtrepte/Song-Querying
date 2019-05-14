import os.path as osp, time, os

from parameters import *

time_stamp = time.strftime('%m-%d_%I-%M-%S')
log_dir = 'finals/' + time_stamp
os.makedirs(log_dir)
log_file = open(osp.join(log_dir, 'log.txt'), 'w')
params_file = open(osp.join(log_dir, 'params.txt'), 'w')

# plot_dir = log_dir + '/plots/'
# os.makedirs(plot_dir)


def print_log(*strings, params=False):
    log_string = ''
    for string in strings:
        log_string += str(string) + ' '
    print(log_string)
    if params:
        params_file.write(log_string+'\n')
    else:
        log_file.write(log_string+'\n')

def flush_file():
	global log_file
	log_file.flush()

def close_file():
	global log_file
	log_file.close()
