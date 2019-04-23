LR = .01
beta1 = .9
beta2 = .999
decay_LR = True
decay_LR_step = 2000

dropout_rate = .0
L2_reg = 0.000

batch_size = 256
num_classes = 5

num_steps = 100000
display_step = 25
save_step = 100

train_percentile = .8

# These don't seem to work
log_spectrograms = False
standard_normalize = False


blacklist = set(['blacklist'])
file_vars = [(item, globals()[item]) for item in dir() if not item.startswith("__")]
desc = ''
for name, val in file_vars:
    if name not in blacklist:
        desc += name + ' = '  + str(val) + '\n'