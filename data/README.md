## `dog{xxx}.npz` files
The files are in `numpy` format. They contain the image, and features
from blocks 1, 2, 3, 4.

To **load**, do:

```python
loaded = np.load('data/dog000.npz')
image = loaded['image']
b1 = loaded['b1']
b2 = loaded['b2']
b3 = loaded['b3']
b4 = loaded['b4']
```

Model used was resnet50 `torchvision.models.resnet.ResNet` with
`pretrained=True`. I think this is standard supervised trained on
imagenet. Check https://pytorch.org/vision/0.12/models.html for more info.

Code for saving is:

```python
def resnet_blocks(resnet: torchvision.models.resnet.ResNet, x: torch.Tensor) -> List[torch.Tensor]:
    with torch.no_grad():
        res = []
        x = resnet.conv1(x)
        x = resnet.bn1(x)
        x = resnet.relu(x)
        x = resnet.maxpool(x)

        x = resnet.layer1(x)
        res.append(x)

        x = resnet.layer2(x)
        res.append(x)

        x = resnet.layer3(x)
        res.append(x)

        x = resnet.layer4(x)
        res.append(x)
        
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
        return [b.permute(0, 2, 3, 1) for b in res]

imgs_blocks = inference(img_testloader, model, DEVICE)
for i, (img, blocks) in enumerate(imgs_blocks):
    fname = f'./data/dog{i:0>3}.npz'
    print(f"Saving to: {fname}")
    print("Dims are: ")
    for b in blocks:
        print(b.shape)
    np.savez_compressed(fname, image=img, 
                        b1=blocks[0], 
                        b2=blocks[1], b3=blocks[2], b4=blocks[3])
```

The current dogs are created from the val set of https://drive.google.com/drive/folders/1Ycq9ZU7I1tIavli_aoV-nayMIhkYlovG.
