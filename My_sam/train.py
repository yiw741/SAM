import torch
from torch import optim, nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from My_test.My_sam.image_encoder import ImageEncoderViT

from My_test.My_sam.prompt_encoder import PromptEncoder
from dataset import CustomDataset
from sam import Sam  # 确保导入正确的SAM模型
from helper_more import eval_metrics
import numpy as np
import os

from My_test.My_sam.mask_decoder  import MaskDecoder
from My_test.My_sam.transformer import TwoWayTransformer

# 设备选择
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 配置参数
data_root = r'D:\data_demo\demo_voc'
save_dir = r'./weights'
os.makedirs(save_dir, exist_ok=True)

EPOCH = 200
num_classes = 21
batch_size = 4
pre_val = 2
crop_size = 256

# 数据加载
train_datasets = CustomDataset(
    root=data_root,
    split='train',
    num_classes=num_classes,
    base_size=300,
    crop_size=crop_size,
    scale=True,
    flip=True,
    rotate=True
)
val_datasets = CustomDataset(
    root=data_root,
    split='val',
    num_classes=num_classes,
    base_size=300,
    crop_size=crop_size,
    scale=True,
    flip=True,
    rotate=True
)

train_dataloader = DataLoader(
    train_datasets,
    batch_size=batch_size,
    num_workers=1,
    shuffle=True,
    drop_last=True
)
val_dataloader = DataLoader(
    val_datasets,
    batch_size=batch_size,
    num_workers=1,
    shuffle=True,
    drop_last=True
)

# 初始化 SAM 模型的各个组件
def build_sam_model(num_classes):
    # 根据 SAM 模型的实际初始化方式调整
    image_encoder = ImageEncoderViT()         # 需要根据实际情况填写
    prompt_encoder = PromptEncoder(embed_dim=256,            # 嵌入维度
                                   image_embedding_size=(64, 64),  # 图像嵌入大小
                                   input_image_size=(1024, 1024),  # 输入图像尺寸
                                   mask_in_chans=16       )   # 掩码通道数)   # 需要根据实际情况填写
    mask_decoder = MaskDecoder(
        transformer_dim=256,
        transformer=TwoWayTransformer,
        num_multimask_outputs=3,
        activation=nn.GELU,
        iou_head_depth=3,
        iou_head_hidden_dim=256
    )

    model = Sam(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder
    )
    return model

# 实例化模型
model = build_sam_model(num_classes)

# 优化器和学习率调度
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=50,
    gamma=0.1
)

# 损失函数
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_epoch(epoch):
    total_loss = 0
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0

    model.to(device)
    model.train()

    tbar = tqdm(train_dataloader, ncols=130)
    for index, (image, label) in enumerate(tbar):
        image = image.to(device)
        label = label.to(device)

        # 准备 SAM 模型的输入
        batched_input = [
            {
                "image": img,
                "original_size": img.shape[-2:],
                # 如果有点提示或其他提示，在这里添加
            } for img in image
        ]

        # SAM 前向传播
        outputs = model(batched_input, multimask_output=False)

        # 从输出中提取掩码
        pred_masks = torch.stack([out['masks'].squeeze(1) for out in outputs])

        # 计算损失
        loss = loss_fn(pred_masks, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()
        lr = get_lr(optimizer)

        total_loss += loss.item()

        # 评估指标计算
        seg_metrics = eval_metrics(pred_masks, label, num_classes)
        correct, num_labeled, inter, union = seg_metrics

        total_correct += correct
        total_label += num_labeled
        total_inter += inter
        total_union += union

        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()

        tbar.set_description(
            'TRAIN {}/{} | Loss: {:.3f}| Acc {:.2f} mIoU {:.2f} | lr {:.8f}|'.format(
                epoch, EPOCH,
                np.round(total_loss/(index+1),3),
                np.round(pixAcc,3),
                np.round(mIoU,3),
                lr
            )
        )

    return total_loss / (index + 1)

# val_epoch 函数类似修改

def train(EPOCH):
    print(f'正在使用 {device} 进行训练! ')
    for i in range(EPOCH):
        train_loss = train_epoch(i)
        if i % pre_val == 0:
            val_loss, val_metrics = val_epoch(i)

if __name__ == '__main__':
    train(EPOCH)
