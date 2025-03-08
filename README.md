# SAM



## 理论实现

### token

token中存放的是当前所需查询的物体，由位置编码传进需求，假如由100个token那么，如果点击位置产生的位置编码分类别进入token的类别没那么多，则其他token会摸鱼

token中是所需要分割的类别，然后在原图中一个一个进行寻找



### 数据引擎

半自动+半手动

手动阶段：

- 那现有已经标好的小数据集训练一个粗糙版的SAM
- 粗糙版的SAM分割新数据人工修正后，重新训练模型
- 上两步反复迭代

半自动阶段：

用训练好的SAM模型，分割图像把分割不出来的给人，人来标，重新训练模型

全自动阶段：

SAM模型在图像上分割，对得分较高，准确率较高，符合设定规则的结果进行保留



## 代码

### Image Encoder

Image Encoder提取给定图像的图像**嵌入向量**。

ImageEncoderViT的主要实现代码如下，其先由patch_embed对输入数据进行16倍的下采样（将patch_size设为16）;并将embed_dim[token长度]设为768（原始的vit中embed_dim是为196=img_w*img_h/(patch_w*patch_h)，img_w:224 patch_w:16）,这与原始ViT相比，是存在一定信息缺失的（64x64=4096）。

**在打成batch时进行的绝对位置编码一次，在tansformer中进行的相对位置编码（每一次循环进行一次）**

**运行过程**

- 在Image Encoder中，先使用PatchEmbed这个类构建将图像转化为patch的Embedding层。之后判断是否使用绝对位置嵌入，也就是增加位置信息，若需要则将一个实现的层加入进pos_embed这个列表中，供之后使用。

- 上一步进行位置嵌入等前置条件后，循环调用block类进行窗口注意力和残差传播，也就是开始进行tansformer,其中核心主要是attention类的运行，最后经过MLPBlock。循环结束后经过neak层的压缩进行输出，这时的输出的矩阵上的值是符合主观语义的值

  - **block中的操作：**在进行attention的前后会进行**`window_partition` 和 `window_unpartition`**，其用于在PyTorch中实现窗口分割和逆分割操作，也就是还原。

    - ![img](https://i-blog.csdnimg.cn/blog_migrate/869f0a8e5ba1071449952b176850a549.png#pic_center)
    - ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/09137c9d9d1bdea05878d90aa17c8f94.png#pic_center)
    - 这些函数主要用于处理图像数据，将其划分为较小的窗口进行处理，然后再将窗口还原为原始图像。
    - 这些每一个小窗口都会完整进行多头attention。

  - **attention类**：实现的多头注意力，

    ```
    head_dim = dim // num_heads
    ```

    会将一层的qkv再次进行分层为每个维度为head_dim的num_heads个qkv，以对应头的形式进行计算

    ```python3
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    ```

    通过线性层将输入 `x` 映射到查询（Q）、键（K）和值（V）的空间，输出形状为 `(B, H, W, 3 * C)`，`reshape` 和 `permute`：将输出重塑并排列成 `(3, B, num_heads, H * W, head_dim)`，其中 `head_dim` 是每个注意力头的维度。

    ```
    q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
    ```

    `unbind(0)`：在第一个维度上分离出查询、键和值，每个的形状为 `(B * num_heads, H * W, head_dim)`。

    ```
    attn = (q * self.scale) @ k.transpose(-2, -1)
    ```

    计算注意力分数

    ps:在多头注意力中还会有一个相对位置

    ```
    x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
    ```

    加权求和

    以下两幅图互补

![image-20241115175438756](SAM.assets/image-20241115175438756.png)

![image-20241115180310051](SAM.assets/image-20241115180310051.png)

![img](https://i-blog.csdnimg.cn/blog_migrate/721d65bffe3543a7e87b970a4a15eb54.png#pic_center)



MLP（多层感知机，Multi-Layer Perceptron）是神经网络中的一种基本结构，由多个全连接层（也称为密集层）组成。通常由输入层、一个或多个隐藏层和输出层组成。

在 Vision Transformer（ViT）等现代架构中，MLP 层通常用于在注意力机制之后进一步处理和转换特征表示。通过将特征维度扩展到更高的维度（由 `mlp_ratio` 决定），MLP 层能够捕捉到更复杂的特征关系。



为什么在进行绝对位置编码时的批次数固定为1

```
def pos_embed_explanation():

 # 位置编码描述的是图像的空间结构
 # 不管输入多少张图像，空间网格是相同的
# 例子：224x224 图像，16x16 patch
# 无论处理 1 张还是 32 张图像
# 空间网格始终是 14x14

# 位置编码捕捉的是：

# 1. 空间位置信息

# 2. 不依赖具体图像内容

# 3. 可以被所有图像复用
```













###  **Positional Encoding**

PromptEncoder用于对输入模型的points、boxes和masks信息进行编码，将其统一为空间特征编码的格式。
其编码器并不复杂，属于轻量化的结构。其对points、boxes和masks编码时允许有部分值空缺（空缺使用默认值），其将points和boxes组装为sparse_embeddings，将mask组装为dense_embeddings`其对mask的采样由多个attention层实现`

PromptEncoder对于points和boxes都使用**PositionEmbeddingRandom**（pe_layer）进行编码，可以看到其将boxes转换为2个点然后输入了模型。point输入，则对应着pe_layer输出的【2和3】

![image-20241113214253939](SAM.assets/image-20241113214253939.png)







Embed_Points

将点坐标和标签进行嵌入处理，包括添加额外的“非点”坐标以填充或表示特殊情况。

![img](https://i-blog.csdnimg.cn/blog_migrate/e15e8e2010a6cb05da0053812a7ab410.png)

**`points`**: 输入的点坐标，形状为 `(B, N, 2)`，其中 `B` 是批量大小，`N` 是每张图像中的点数，`2` 表示坐标的两个维度（通常是 x 和 y）。

**属性labels**： 每个点对应的标签，形状为 `(B, N)`，标签值通常表示点的类别（如背景、目标等）。

```python
points = points + 0.5
```

- 将点坐标移动到像素中心，points一开始都会指向当前像素块的左上角，当i,j都加0.5就会移至中央



```
point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
self.point_embeddings = nn.ModuleList(point_embeddings)
```

- `point_embeddings` 被创建为一个包含多个 `nn.Embedding` 实例的列表，并通过 `nn.ModuleList` 将其转换为一个可被训练的模块列表。这种方式常用于在模型中创建多个嵌入层，每个嵌入层可以独立地学习特定类型的输入特征。self.num_point_embeddings 是需要的嵌入层数量，默认为4

  - 4代表了表示pos/neg + 2 box corners，即demo里面的添加点和消除点、以及box框的左上角和右下角；

  - "4"表示的是提示的**种类**而不是数量

    - 0：neg，对应demo中的消除点

      1：pos，对应demo中的添加点

      2：代表box左上角点

      3：代表box右下角点

  - 意味着用户可以提供不同类型的提示来引导模型进行更准确的分割。每种类型的提示可以有多个实例，例如多个正点或负点。

  - 意味着这四种类型的点，每个类型里面可以有多种实例

- 正点（Positive Point）和负点（Negative Point）在SAM中的使用通常是通过用户交互来实现的。这些点是用户在图像上手动标记的，用于引导模型进行更准确的分割。

```
self.not_a_point_embed = nn.Embedding(1, embed_dim)
```

- **再生成一组可学习的向量not_a_point_embed，大小为1x256，用于表示该位置不是一个点**



```
if pad:            
    padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)            
    padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)            
    points = torch.cat([points, padding_point], dim=1)            
    labels = torch.cat([labels, padding_label], dim=1)
```

- **如果传入的prompt里面没有bbox，则补充一个【0，0】点到每个point后面，其对应的label为-1，此时point大小为Nx2x2，label为Nx2**

  - 输入数据通常需要有固定的格式和结构。通过在每个点后面补充一个【0,0】点，确保即使在缺少边界框的情况下，输入数据的格式仍然一致。

  - 标签为-1的点通常被视为无效点
  - 这些无效点是额外添加的，不会替代或改变原始输入的有效点。
  - **pad：**判断当前输入是否存在输入框，若不存在，则会为falus进行无效点的制造
  - **作用1：**模型可能设计时就默认存在输入框，但实际操作时并没有输入框。边界框通常用于限定点的范围或作为上下文信息的一部分。如果缺少边界框，可能会导致输入数据结构不完整，从而影响模型的处理
  - **作用2：**在批处理模式下，所有输入样本通常需要具有相同的结构和维度。如果某些样本没有边界框，而其他样本有，这可能导致输入张量的形状不一致，从而使批处理操作变得复杂。
  - **总结：**增加无效点取代box

- **如果传入的还有bbox，此时的point大小为Nx1x2，label为Nx1**

坐标值的大小在[0,1]，但其是（1，256）维的



### MaskDecoder

用于根据PromptEncoder和ImageEncoderViT的输入生成mask，其是一种transformer架构的解码头，有两个输出头：output_upscaling和iou_prediction_head，两个头输出的数量是一样的。
其forward函数仅是对输出结果进行了选择操作，核心推理是由predict_masks函数和transformer对象（TwoWayTransformer实例）完成的

**TwoWayTransforme:**该Transformer对象承担了MaskDecoder在预测iou和mask时的绝大部分工作，其预测时需输入image_embedding、image_pe、point_embedding(，预测完输出queries（在后续中被作为iou_token和mask_token）, keys（在后续中被作为mask）

mask提示允许我们直接在原图上指示感兴趣区域来引导模型。这些mask通过卷积操作被转换为与图像嵌入空间相匹配的特征，然后与图像嵌入相加结合，为模型提供分割的精确位置信息。

如果没有使用mask提示，则将一组可学习向量(no_mask_embed,1*256)expand为1x256×64×64后替代，使得在处理序列数据时，即使没有具体的mask信息，也能有一个统一的处理方式。

**predict_masks**:MaskDecoder中核心函数，用于实现mask和iou的预测。

1. **概率值输出**：
   - SAM生成的掩码通常以概率值的形式输出，这些值表示每个像素属于目标对象的概率。
   - 这种输出形式允许模型在不确定的情况下表达不同程度的置信度。

![img](https://i-blog.csdnimg.cn/blog_migrate/90b8ec9af1db1d411d371349e77b07fe.png#pic_center)

![image-20241116003722622](SAM.assets/image-20241116003722622.png)

![image-20241201120525066](../../../../Typora/image/image-20241201120525066.png)



### TwoWayTransformer

![img](https://i-blog.csdnimg.cn/blog_migrate/fcc9c004b13f069d229b40c40aa38132.png)

<img src="../../../../Typora/image/image-20241201120542261.png" alt="image-20241201120542261" style="zoom:80%;" />



## 训练

这是训练过程的一个示例：

1. 模型从 `images/` 文件夹中读取一个图像，例如 `image1.jpg`。
2. 模型从 `annotations/` 文件夹中读取相应的标注文件，例如 `image1.json`。
3. 模型使用标注文件生成图像的一个框或掩码，例如一个点或框指示要分割的对象。
4. 模型处理图像并生成一个分割掩码基于框或掩码。
5. 模型计算预测分割掩码和ground-truth掩码（如果可用）之间的损失函数。
6. 模型根据损失函数和反向传播更新其参数。

```
dataset/
├── images/
│   ├── image1.jpg          # 原始输入图像
│   ├── image2.jpg
│   └── ...                 # 其他图像
├── masks/
│   ├── image1_mask.png     # 图像对应的掩码（分割结果）
│   ├── image2_mask.png
│   └── ...                 # 其他掩码
├── annotations/
│   ├── image1.json         # 图像的注释文件（如有）
│   ├── image2.json
│   └── ...                 # 其他注释
```

**掩码：**假设您有一张照片 `image1.jpg`，其掩码文件 `image1_mask.png` 具体内容如下：（掩码是）

- **原始图像**：可能包含多个对象，比如行人、汽车和树木。
- 掩码（image1_mask.png）
  - 背景: 像素值 0
  - 行人: 像素值 1
  - 汽车: 像素值 2
  - 树木: 像素值 3

在这种情况下，通过加载掩码图像，可以轻松得到每个对象的分割区域，进而进行分析、训练或推理。
