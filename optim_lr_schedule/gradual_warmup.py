import numpy as np
warmup_steps=2500
init_lr=0.1
max_steps=15000
for train_steps in range(max_steps):
    if warmup_steps and train_steps<warmup_steps:
        warmup_percent_done=train_steps/warmup_steps
        warmup_learning_rate=init_lr*warmup_percent_done
        learning_rate=warmup_learning_rate
    else:
        learning_rate=learning_rate**1.0001
    if (train_steps+1)%100==0:
        print('train_step:%.3f--warmup_step:%.3f--learning_rate:%.3f'%(train_steps+1,warmup_steps,learning_rate))