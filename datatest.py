import torchvision

dataset = torchvision.datasets.ImageFolder('E:\py_pro\data') # 不做transform
print(dataset.classes)
print(dataset.class_to_idx)
print(dataset.imgs[0])