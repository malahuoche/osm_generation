# 实际算法与范围

## 主路径

RGB + 标定 → 官方 MIA image_encoder → scale_classifier → polar projection → Cartesian BEV（128×100×100）。缓存的是裁去末列后的原始特征；生成模型使用训练集支持区域计算的逐通道均值/标准差，裁剪标准化值到[-8,8]后乘投影支持。

新生成器：128→16通道投影；16/24/32通道条件金字塔；时间FiLM残差U-Net；25×25瓶颈空间交叉注意力。没有辅助语义头。相机位置/网格坐标是固定局部坐标，不输入GPS或GT地图。

## 三种生成

- **x0**：目标 `m=2*y-1`。1000步cosine噪声日程；训练预测干净m。推理DDIM eta=0，clip clean到[-1,1]后重算相同clean对应的噪声，避免不一致采样。
- **FM**：`z_t=(1-t)*noise+t*m`，预测速度 `m-noise`；推理Euler积分0→1。训练边界项使用单步端点估计，而不是假称完整ODE输出。
- **latent_x0**：连续AE把6×100×100地图编码为12×50×50；按训练地图latent统计归一化后训练x0 diffusion。去噪器在100×100条件网格计算，latent上采样后输入、预测平均池化回50×50。连续latent没有[-1,1]裁剪。最后通过冻结AE解码。此设计测试表示变化，不宣称计算加速。

损失为生成MSE + 0.1边界项；边界项对道路/行人区域/建筑的signed-map相邻梯度计算SmoothL1，按0.5米换算，评估支持边缘内缩1米。x0按alpha_bar、FM按t²加权。latent边界梯度通过冻结的AE解码器回传到生成器。

AE独立训练：BCE + 0.5 softDice + 0.1地图梯度L1；不存在GT encoder skip通道。AE重建门槛与生成质量是两件事。

## 与 DiffMap 的区别

参考 [DiffMap](https://arxiv.org/html/2405.02008v2) 的地图latent条件扩散思路。本项目使用连续AE，而非其VQ-VAE；没有其epsilon/clean-latent双分支、实例embedding/方向头、原采样器及三次采样平均。因此必须称为DiffMap-inspired，不是完整复现。

## 冻结与解冻

服务器主入口读取固定BEV，因此图像骨干、图像投影层、尺度头都被冻结。更新后面的条件适配器不等于解冻MIA。历史scale-head小试验的结果只记录在实验表；要研究全骨干联合训练，必须新增在线RGB路径、正确保存上游参数并建立匹配的冻结对照，不能继续把BEV缓存当可训练输入。
