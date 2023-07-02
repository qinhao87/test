import torch
import visdom

vis = visdom.Visdom(env='first')
vis.text('first visdom', win='text1')
vis.text('hello pytorch', win='text1', append=True)

for i in range(20):
    vis.line(X=torch.FloatTensor([i]), Y=torch.FloatTensor([i]),
             opts={'title': 'y=x'}, win='x=y', update='append')

vis.image(torch.randn(3, 256, 256), win='image')
