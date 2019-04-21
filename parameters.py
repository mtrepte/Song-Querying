LR = .01
beta1 = .9
beta2 = .999
decay_LR = True
decay_LR_step = 1000

dropout_rate = .3
L2_reg = 0.001

batch_size = 128
num_classes = 5

num_steps = 100000
display_step = 25
save_step = 100

train_percentile = .8



blacklist = set(['blacklist'])
file_vars = [(item, globals()[item]) for item in dir() if not item.startswith("__")]
desc = ''
for name, val in file_vars:
    if name not in blacklist:
        desc += name + ' = '  + str(val) + '\n'