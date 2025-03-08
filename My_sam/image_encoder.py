
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from common import LayerNorm2d, MLPBlock

class ImageEncoderViT(nn.Module):
    def __init__(
            self,
            img_size: int = 1024,  # 输入图像的大小（假设为正方形），默认1024
            patch_size: int = 16,  # 图像分割成补丁的大小，默认16
            in_chans: int = 3,  # 输入图像的通道数，通常为3（RGB），默认3
            embed_dim: int = 768,  # 嵌入维度，每个补丁被嵌入到一个768维的向量中
            depth: int = 12,  # Transformer编码器的层数，决定编码器的深度，默认12
            num_heads: int = 12,  # 多头自注意力机制中的头数，默认12
            mlp_ratio: float = 4.0,  # MLP隐藏层的扩展比率，用于调整隐藏维度，默认4.0
            out_chans: int = 256,  # 输出通道数，决定编码器输出的通道数，默认256
            qkv_bias: bool = True,  # 查询、键、值向量中是否使用偏置，默认True
            norm_layer: Type[nn.Module] = nn.LayerNorm,  # 归一化层的类型，默认使用LayerNorm
            act_layer: Type[nn.Module] = nn.GELU,  # 激活函数的类型，默认使用GELU
            use_abs_pos: bool = True,  # 是否使用绝对位置编码，默认True
            use_rel_pos: bool = False,  # 是否使用相对位置编码，默认False
            rel_pos_zero_init: bool = True,  # 是否将相对位置编码初始化为零，默认True
            window_size: int = 0,  # 窗口大小，可能用于局部注意力机制，默认0
            global_attn_indexes: Tuple[int, ...] = (),  # 全局注意力的索引，默认空元组
    ) -> None:

        super().__init__()
        self.img_size = img_size
        #PatchEmbed 类通常用于将输入图像分割成多个小图像块（patches）
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim, # 嵌入维度768
                num_heads=num_heads, # 多头注意力机制heads=12
                mlp_ratio=mlp_ratio, # mlp隐藏层的维度变化因子4
                qkv_bias=qkv_bias, # qkv全连接层的偏执True
                norm_layer=norm_layer, # 归一化层，nn.LayerNorm
                act_layer=act_layer, # 激活函数层 nn.GELU
                use_rel_pos=use_rel_pos, # 是否添加相对位置嵌入False
                rel_pos_zero_init=rel_pos_zero_init, # 零初始化相对位置嵌入参数True
                # sam_vit_b中的global_attn_indexes=encoder_gloabl_attn_indexed=[2,5,8,11]
                # 12个Block中的window_size[14,14,0,14,14.0,14,14,0,14,14,0]
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size), # 输入大小(64,64)
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    #这个类的执行流程
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #进行判断当前的通道数大小，如果为1，则将其转换为三通道
        if x.size()[1]==1:
            #3表示在通道维度上重复3次，1表示在高度和宽度维度上不重复。因此，x.repeat(1,3,1,1)会将一个单通道的灰度图像转换为三通道的彩色图像。
            x = x.repeat(1,3,1,1)

        #进行分块降维处理
        x = self.patch_embed(x)

        #进行位置编码，也就是嵌入
        if self.pos_embed is not None:
            x = x + self.pos_embed
        #以上进行了位置嵌入等前置
        #进行12个block的循环处理，也就是tansformer的12层处理
        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class Block(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        #进行的MLP
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x # [1,64,64,768]

        #进行均值归一化
        x = self.norm1(x)

        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2] # H=64,W=64
            x, pad_hw = window_partition(x, self.window_size) # # x.size():[25, 14, 14, 768], Pad_hw.size():[70, 70]

        #进行多头注意力
        x = self.attn(x) # [25, 14, 14, 768]

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W)) # [1, 64, 64, 768]

        #最开始输入，与现在进行多头注意力之后的残差计算
        x = shortcut + x # 残差连接

        #进行MLP层计算
        x = x + self.mlp(self.norm2(x)) # [1, 64, 64, 768]

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:

        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads#头维度
        self.scale = head_dim**-0.5
        #qkv的计算，只是一层全连接层,也就是输出q,,k,v
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        #
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."

            #相对位置编码：如果启用相对位置编码，self.rel_pos_h 和 self.rel_pos_w 分别初始化为高度和宽度方向的相对位置参数。这些参数用于在注意力计算中加入相对位置信息。
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, H, W, _ = x.shape

        # 输入 x 的形状是 (B, H, W, C)，经过线性变换后，输出的形状变为 (B, H, W, 3 * C)
        #通过 self.qkv(x) 生成查询、键和值的组合，形状为 (B, H * W, 3 * dim)。
        #将其重塑为 (B, H * W, 3, num_heads, head_dim)，然后通过 permute 调整为 (3, B, num_heads, H * W, head_dim)
        # 使得Q、K、V分离。
        #三表示的是q,k,v的类别数量
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   #-1 自动推断每个头的特征维度（head_dim）dim // num_heads

        # q, k, v with shape (B * nHead, H * W, C)
        #unbind(0) 将第一个维度（即表示Q、K、V的维度）拆分开来，分别得到三个张量：q、k、v。
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        # q.size():[6, 25, 768], k.size():[6, 25, 768], v.size():[6, 25, 768]

        #进行注意力计算，矩阵运算，transpose进行维度交换，即倒置
        attn = (q * self.scale) @ k.transpose(-2, -1)

        #是否使用相对路径运算
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        #与v进行计算
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

#进行分割窗口
def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:

    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

#拼接窗口
def window_unpartition(
        windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:

    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

#计算查询和键之间的相对位置编码。
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:

    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

#将相对位置编码加入到注意力权重中。使用get_rel_pos方法
def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
) -> torch.Tensor:

    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
            attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


#进行了一次卷积用于将原图进行分割成小块也就是降维，打成patch
class PatchEmbed(nn.Module):
    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16),
            stride: Tuple[int, int] = (16, 16),
            padding: Tuple[int, int] = (0, 0),
            in_chans: int = 3,
            embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
