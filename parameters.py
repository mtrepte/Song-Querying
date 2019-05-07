LR = .01
beta1 = .9
beta2 = .999
decay_LR = True
decay_LR_step = 2500

threshold = .5

dropout_rate = .4
L2_reg = 0.0001

batch_size = 128
num_classes = 30

num_steps = 100000
display_step = 25
save_step = 2500

train_percentile = .8

train_with_usage_embs = True
usage_loss_weight = 100.

embedding_dim = 128

# These don't seem to work
log_spectrograms = False
standard_normalize = False


blacklist = set(['blacklist'])
file_vars = [(item, globals()[item]) for item in dir() if not item.startswith("__")]
desc = ''
for name, val in file_vars:
    if name not in blacklist:
        desc += name + ' = ' + str(val) + '\n'