model = 'SiamRPN'
original_lr = 1e-3
lr = 1e-3
batch_size = 8
momentum = 0.9
decay = 5 * 1e-4
start_epoch = 0
epochs = 400
steps = [-1, 1, 40, 60, 70]
scales = [.1, 10, .1, .1, .1]
workers = 4
seed = 100
print_freq = 1
test_len = 300
data_type = 'RPN'







