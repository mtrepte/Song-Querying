LR = .001
beta1 = .9
beta2 = .999
decay_LR = True
decay_LR_step = 5000

stddev = .1
threshold = .25

dropout_rate = 0.2
L2_reg = 0.0001

batch_size = 256
num_classes = 10

num_steps = 100000
display_step = 25
save_step = 2500

train_percentile = .8

train_with_usage_embs = True
usage_loss_weight = 10.

embedding_dim = 128

# These don't seem to work
log_spectrograms = True
standard_normalize = True


blacklist = set(['blacklist'])
file_vars = [(item, globals()[item]) for item in dir() if not item.startswith("__")]
desc = ''
for name, val in file_vars:
    if name not in blacklist:
        desc += name + ' = ' + str(val) + '\n'
