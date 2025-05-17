import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models

# レイヤー指定
CONTENT_LAYERS = ['conv4_2']
STYLE_LAYERS   = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']

def get_image(path, size):
    img = Image.open(path).convert('RGB')
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Lambda(lambda x: x.mul(255))
    ])
    return transform(img).unsqueeze(0)  # (1,3,H,W)

def gram_matrix(feat):
    b, c, h, w = feat.size()
    f = feat.view(b, c, h*w)
    return torch.bmm(f, f.transpose(1,2)) / (c*h*w)

class StyleTransferNet(nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.model  = cnn.features.eval()
        # 各レイヤーの出力を取り出せるよう登録
        self.content_idxs = []
        self.style_idxs   = []
        for i, layer in enumerate(self.model):
            name = f"conv{i//2+1}_{i%2+1}"  # 例: conv4_2
            if name in CONTENT_LAYERS: self.content_idxs.append(i)
            if name in STYLE_LAYERS:   self.style_idxs.append(i)

    def forward(self, x):
        content_feats = []
        style_feats   = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.content_idxs:
                content_feats.append(x)
            if i in self.style_idxs:
                style_feats.append(x)
        return content_feats, style_feats

import datetime

def main():
    content = "img/cat.jpg"
    style = "styles/sunset.jpg"
    now = datetime.datetime.now()
    output = f"output/{now.year}年{now.month}月{now.day}日_{now.hour}時{now.minute}分{now.second}秒.jpg"
    size = 400
    iters = 1000
    content_weight = 100
    style_weight = 1e4

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"device: {device}")
    content_img = get_image(content, size).to(device)
    style_img   = get_image(style,   size).to(device)
    input_img   = content_img.clone().requires_grad_(True)

    cnn = models.vgg19(pretrained=True).to(device).eval()
    net = StyleTransferNet(cnn)

    with torch.no_grad():
        target_content, _ = net(content_img)
        _, target_styles  = net(style_img)
        target_grams      = [gram_matrix(f) for f in target_styles]

    optimizer = optim.LBFGS([input_img], lr=1.0)

    run = [0]
    while run[0] < iters:
        def closure():
            optimizer.zero_grad()
            content_feats, style_feats = net(input_img)
            content_loss = 0
            for f, t in zip(content_feats, target_content):
                content_loss += nn.functional.mse_loss(f, t)

            style_loss = 0
            for f, g in zip(style_feats, target_grams):
                style_loss += nn.functional.mse_loss(gram_matrix(f), g)

            loss = content_weight * content_loss + style_weight * style_loss
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Iter {run[0]}/{iters}, Content: {content_loss.item():.2f}, Style: {style_loss.item():.2f}")
            return loss

        optimizer.step(closure)

    output_img = input_img.detach().cpu().squeeze().clamp(0,255).div(255)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    T.ToPILImage()(output_img).save(output)
    print("▶ Saved:", output)

if __name__ == "__main__":
    main()
