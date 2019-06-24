model = 'SiamRPNPP'
original_lr = 1e-3
lr = 1e-3
start_epoch = 0
backbone_train_epoch = 150
warmup = True
lr_warm_epoch = 10
batch_size = 16
momentum = 0.9
decay = 0.0001
start_epoch = 0
epochs = 400
steps = [-1, 1, 40, 60, 70]
scales = [.1, 10, .1, .1, .1]
workers = 4
seed = 100
print_freq = 1
test_len = 300
data_type = 'RPN'







