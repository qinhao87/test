import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

learning_rate = 1.
num_epochs = 50

weight = torch.randn((1,), dtype=torch.float32)
print(weight)
optimizer = SGD([weight], lr=learning_rate, momentum=0.9)
print(optimizer)
scheduler = StepLR(
    optimizer=optimizer,
    step_size=10,
    gamma=0.1,
    last_epoch=-1
)
print(scheduler)
lrs,epochs=[],[]
for epoch in range(num_epochs):
    lrs.append(scheduler.get_lr())
    epochs.append(epoch)
    pass

    scheduler.step()
print(lrs,epochs)
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.plot(epochs,lrs,label='StepLR')
plt.legend()
plt.show()