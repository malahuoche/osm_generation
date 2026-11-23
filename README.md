# OSM Generation：MIA BEV 条件地图生成

把前视 RGB 转为局部 OSM 风格语义栅格的研究工程。当前主线：**冻结的 MIA BEV 特征 → x₀ diffusion / Flow Matching / 地图 latent x₀ diffusion → 六层语义栅格**。提供原始 MIA mapper 对照；不使用 LiDAR、SIFT，也不把 MIA 的初步地图作为生成模型输入。

这是此前本地实验的可迁移整理版，不是 MIA / DiffMap 官方复现。训练目标、网络和推理步骤见 [算法说明](docs/architecture.md)。历史小规模结果见 [实验记录](docs/pilot-results.md)。

## 1. 当前支持范围

- CUDA / CPU / MPS 单设备、FP32；数据按样本从磁盘读取，不一次性装入全部特征。
- x₀、FM、地图 AE、latent x₀ 的训练、验证集选模型、断点续训、独立测试、预测文件导出。
- 官方 MIA 数据索引、RGB→BEV 缓存、已有 KITTI 配对实验导出。
- 六个**独立二值层**，不是 softmax 互斥分类：road / crossing / sidewalk（官方 explicit_pedestrian）/ building / terrain / parking。
- 固定 100×100、0.5 m/格、50×50 m；相机连续像素坐标 `(row=100,col=50)`，前方朝上。不能只修改图像大小就改变地图范围。
- **当前服务器训练入口冻结整个 MIA 特征提取器**。缓存不能用于解冻 DINO；历史的“仅尺度头200步”实验不是全骨干微调。本版本不声称已支持端到端解冻、DDP 多卡、AMP 或时序融合。

大规模是指可读取更多样本、训练更多步；不是已经完成百万样本训练或 CUDA 大规模性能验证。先完成下面的小样本检查，再扩展数据量。

## 2. 服务器安装

在 Linux NVIDIA 服务器上，以 Python 3.11 为例：

```bash
git clone https://github.com/malahuoche/osm_generation.git
cd osm_generation
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[mia]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

必须确认最后输出 CUDA 可用。上述 CUDA 12.1 wheel 是一套固定参考组合；按服务器驱动选择 [PyTorch 官方历史版本安装表](https://pytorch.org/get-started/previous-versions/)。不要混用不匹配的 torch / torchvision。只有预先缓存好的数据时，`pip install -e .` 即可，不需要 MIA 可选依赖。

### MIA 官方代码与预训练权重

```bash
mkdir -p third_party weights
git clone https://github.com/MapItAnywhere/MapItAnywhere.git third_party/MapItAnywhere
git -C third_party/MapItAnywhere checkout 6dbfaa569c7f384204a9c4c5b915414c1c87451f
export MIA_REPO="$PWD/third_party/MapItAnywhere"
curl -L --fail --retry 3 \
  https://huggingface.co/mapitanywhere/mapper/resolve/main/weights/mapper-excl-ood/model.ckpt \
  -o weights/mapper-excl-ood.ckpt
sha256sum weights/mapper-excl-ood.ckpt
```

预期 SHA256：`51a063fa03c24984bcf4bbd15d64011d83f72af209727bc6a5c3caee5aea2735`。代码会验证这个指纹，并严格加载实际使用的参数。只排除作者旧 checkpoint 中、当前官方 decoder 已不存在的三个 confidence-head 张量。下载来源：[作者权重目录](https://huggingface.co/mapitanywhere/mapper/tree/main/weights/mapper-excl-ood)。

`MIA_REPO` 指向**代码目录**，不是数据或权重目录；每个新终端重新设置。`BEV_OSM_CACHE` 可选，默认 `~/.cache/bev-osm`，只用于运行时缓存。不要上传账号 token。

## 3. 数据在哪里下载、放在哪里

**逐步操作见 [数据预处理指南](docs/preprocessing.md)**：已有428张实验的导出和迁移、官方MIA目录索引、RGB/标定/GT的处理、输出检查，以及新KITTI/Boreas尚缺的GT生成步骤。训练读取缓存BEV，不直接读取原始图片。

建议自己设置服务器路径，下面的 `/data/...` 均为示例，代码不硬编码这些位置：

```text
/data/osm-generation/
  mia_raw/                  # 下载解压的原始配对数据
    pittsburgh/
      images/
      semantic_masks/
      flood_fill/
      dump.json
    ...其他城市...
  pairs.jsonl               # RGB、标定、GT、划分索引
  prepared/
    manifest.jsonl          # 训练/测试实际传入的文件
    manifest.meta.json      # 类别和坐标约定
    samples/*.npz           # 原始BEV、GT、支持区、评估区
```

### 方案 A：MIA 已配对数据（建议用于扩展数据量）

- [官方数据说明](https://github.com/MapItAnywhere/MapItAnywhere/blob/main/mia/dataset.md)
- [官方 Box 下载入口](https://cmu.box.com/s/6tnlvikg1rcsai0ve7t8kgdx9ago9x9q)
- [公开少量样例](https://huggingface.co/datasets/cherieho/mia_dataset)：只适合格式检查，不应当成完整训练集。

**下载状态**：2026-11-18 整理时能核对官方说明与权重目录，但未能在当前环境确认 Box 链接可访问；此前本地请求曾返回404。没有把未下载的完整数据写成“已可用”。若链接失效，向作者索取镜像或使用其数据引擎；不要把下载到的 HTML 错当数据压缩包。

解压后编辑 `configs/city_splits.example.json`，使城市名称与真实文件夹一致。示例是自定义整城留出，不是作者官方评测划分；不要把同一地区的相邻帧随机分到训练和测试中。

```bash
python -m bev_osm.index_mia \
  --root /data/osm-generation/mia_raw \
  --splits configs/city_splits.example.json \
  --out /data/osm-generation/pairs.jsonl

python -m bev_osm.prepare \
  --pairs /data/osm-generation/pairs.jsonl \
  --out /data/osm-generation/prepared \
  --mia-checkpoint weights/mapper-excl-ood.ckpt \
  --device cuda

python -m bev_osm.audit \
  --manifest /data/osm-generation/prepared/manifest.jsonl \
  --verify-hashes
```

缓存提取中断后，用相同参数追加 `--resume`；改变配对清单/权重/评估掩膜时必须换输出目录。索引程序报告 dump 中缺失图片/地图的条数，务必检查，不能以少量下载样例冒充完整集。

官方原始地图按中心裁取前方100×100，选原类别 `[0,1,2,4,6,7]`。默认评估范围为相机投影支持区；可加 `--flood-eval-mask` 使用作者 GT-derived flood mask 进一步限制**评估/损失**，但该掩膜绝不输入特征或生成网络。两种评估策略的成绩不能混比。

### 方案 B：官方 MIA 数据引擎

参见 [官方数据引擎步骤](https://github.com/MapItAnywhere/MapItAnywhere#readme)。它需要 Mapillary token 和地理区域配置，用于获得新的配对数据；不是无需凭据的一键完整数据下载。按官方说明在独立 Docker 环境生成城市目录，再使用上面的 `index_mia` 和 `prepare`。本仓库不捆绑它的地理渲染依赖，也不自动修改道路渲染规则。

### 方案 C：KITTI / Boreas

- [KITTI Raw 官方下载](https://www.cvlibs.net/datasets/kitti/raw_data.php)：RGB、标定、位姿；下载可能要求注册。
- [Boreas 官方入口](https://www.boreas.utias.utoronto.ca/)：按其下载说明取得相机、标定和位姿。

**它们的原始 RGB 下载不等于本任务的配对 OSM GT。** 还需要对应区域/日期的 OSM 数据、位姿到相机的变换、与当前栅格一致的渲染。不要拿航拍图、RGB 语义分割图或 LiDAR 标签直接充当本任务 GT。本版本不会自动为任意 KITTI/Boreas 序列生成合格 GT；已有本地 KITTI 渲染结果可直接导出：

```bash
python -m bev_osm.export_local \
  --source /path/to/experiments/mia-v3-locked/data/kitti-native-mia-428 \
  --features /path/to/experiments/mia-minimum/data/latent-osm/raw_features128.npy \
  --out /data/osm-generation/kitti428
python -m bev_osm.audit --manifest /data/osm-generation/kitti428/manifest.jsonl --verify-hashes
```

`--source` 必须含 `metadata.json` 与 `targets.npz`；`--features` 覆盖旧 metadata 中的本机绝对路径。导出后的整个目录可用 `rsync -av --progress <本地导出目录>/ <服务器>:/data/osm-generation/kitti428/` 传到服务器。Git 仓库没有附带这428张数据。

自有配对数据按 `configs/pair.example.jsonl` 制作 JSONL。`target_format=six_chw` 表示单数组 NPZ/NPY 中的6×100×100二值地图；`mia_native` 表示官方多层HWC地图。示例相机参数只是格式示例，必须替换成真实标定；`roll_pitch_degrees` 单位为度。路径相对 pairs.jsonl 所在目录解析，绝对路径也可用。

## 4. 先验证环境，不用下载真实数据

```bash
python -m unittest discover -s tests -v
python -m bev_osm.synthetic --out /tmp/osm-smoke
python -m bev_osm.train --kind x0 --manifest /tmp/osm-smoke/manifest.jsonl \
  --out runs/smoke-x0 --device cpu --workers 0 --batch 2 --steps 2 \
  --validate-every 2 --save-every 1 --val-nfe 2
python -m bev_osm.evaluate --manifest /tmp/osm-smoke/manifest.jsonl \
  --checkpoint runs/smoke-x0/best.pt --out runs/smoke-x0-test \
  --device cpu --workers 0 --nfe 2
```

这是**合成数据的软件测试**，不是论文结果。新实验目录不能覆盖已有 run，续训使用 `--resume`。

## 5. 开始训练

先设置实际数据文件路径：

```bash
export OSM_MANIFEST=/data/osm-generation/prepared/manifest.jsonl
```

### 像素 x₀ 与 Flow Matching

```bash
CUDA_VISIBLE_DEVICES=0 python -m bev_osm.train \
  --kind x0 --manifest "$OSM_MANIFEST" --out runs/x0 \
  --device cuda --batch 8 --workers 4 --steps 100000 \
  --lr 0.001 --boundary 0.1 --validate-every 1000 --save-every 500 --val-nfe 20

CUDA_VISIBLE_DEVICES=0 python -m bev_osm.train \
  --kind fm --manifest "$OSM_MANIFEST" --out runs/fm \
  --device cuda --batch 8 --workers 4 --steps 100000 \
  --lr 0.001 --boundary 0.1 --validate-every 1000 --save-every 500 --val-nfe 20
```

`--steps` 是优化器更新次数，不是 epoch；样本均匀有放回采样。一次近似遍历需要 `训练样本数 / batch` 次更新。示例的10万步不是针对你的大数据已经验证过的最优预算；根据验证曲线、训练集大小和显存调整。

### 地图 AE → latent x₀

```bash
CUDA_VISIBLE_DEVICES=0 python -m bev_osm.train \
  --kind ae --manifest "$OSM_MANIFEST" --out runs/ae \
  --device cuda --batch 16 --workers 4 --steps 20000 \
  --validate-every 1000 --save-every 500
cat runs/ae/ae_gate.json
```

必须检查道路/行人区域/建筑的全图验证 IoU 与边界F1@0.5m均≥0.95。未通过时继续改善 AE；latent 入口会拒绝使用未通过的 codec。`ae/best.pt` 结束时写入**仅训练地图计算**的 latent 均值/标准差。AE 用 GT 做重建是表示能力检查，不是图像→地图的成绩。

```bash
CUDA_VISIBLE_DEVICES=0 python -m bev_osm.train \
  --kind latent_x0 --manifest "$OSM_MANIFEST" --out runs/latent-x0 \
  --ae-checkpoint runs/ae/best.pt \
  --device cuda --batch 8 --workers 4 --steps 100000 \
  --boundary 0.1 --validate-every 1000 --save-every 500 --val-nfe 20
```

AE 解码器冻结，但梯度通过它传回去噪器；没有 encoder→decoder 的 GT 跳接。推理从噪声开始，仅以 MIA BEV 为条件，不读取 GT。

大验证集可设置 `--val-limit 2000`，使用固定等间隔子集选择检查点（默认0表示全量）。这不是完整验证集成绩；最终测试仍读取完整test。续训必须保持相同val-limit。跨数据集使用预训练AE需显式 `--allow-external-codec` 并先检查预训练地域与测试集重叠，默认拒绝不同数据清单的codec。

### 断点续训

```bash
CUDA_VISIBLE_DEVICES=0 python -m bev_osm.train \
  --kind x0 --manifest "$OSM_MANIFEST" --out runs/x0 \
  --resume runs/x0/last.pt --device cuda --batch 8 --workers 4 \
  --steps 200000 --lr 0.001 --boundary 0.1 \
  --validate-every 1000 --save-every 500 --val-nfe 20
```

恢复模型、优化器、更新步数与随机状态；步数指定**最终总步数**。batch、种子、目标类型、学习率、边界权重、验证采样步数必须匹配。保留整个 run 目录（特别是 best.pt），不能只搬 last.pt 到一个空目录。数据清单或样本不能偷偷替换；更换数据需新建实验。跨设备不承诺逐位复现。

## 6. 测试与预测导出

```bash
python -m bev_osm.evaluate --manifest "$OSM_MANIFEST" \
  --checkpoint runs/x0/best.pt --out runs/test-x0 \
  --split test --device cuda --batch 8 --workers 4 --nfe 50 --save-predictions

python -m bev_osm.evaluate --manifest "$OSM_MANIFEST" \
  --checkpoint runs/fm/best.pt --out runs/test-fm \
  --split test --device cuda --batch 8 --workers 4 --nfe 50 --save-predictions

python -m bev_osm.evaluate --manifest "$OSM_MANIFEST" \
  --checkpoint runs/latent-x0/best.pt --out runs/test-latent \
  --split test --device cuda --batch 8 --workers 4 --nfe 50 --save-predictions

python -m bev_osm.evaluate --manifest "$OSM_MANIFEST" \
  --mia-mapper --checkpoint weights/mapper-excl-ood.ckpt --out runs/test-mia \
  --split test --device cuda --batch 8 --workers 4 --save-predictions
```

输出 `metrics.json` 和可选 `predictions/<id哈希>.npz`：其中包含原始 id、六层 continuous 值和 `decoded = continuous >= 0.5`。FM 的 continuous 不强制在[0,1]内，不应当作校准概率；latent 解码输出为 sigmoid 概率。

训练只用 train，选择 best 只用 val；默认测试 test。指标包括类别 IoU、边界F1@0.5m/1m、边界距离、道路 clDice。clDice 是骨架重叠代理，不等于路口拓扑正确；本通用入口尚不包含依赖源OSM矢量的路口支路/建筑实例匹配评估。需要结合固定样例和源矢量检查，不只优化 IoU。

## 7. 大规模数据训练注意事项

- BEV float16 单帧原始大小约2.56 MB，100万帧约2.56 TB（压缩前，未计图片/GT）；优先在独立高速数据盘缓存，并先跑一个完整小子集。仓库不包含TB级数据，也不自动下载全部数据。
- 一样本一NPZ可逐个读取；大量小文件可能成为I/O瓶颈。当前不是 WebDataset/LMDB 分片实现。`--workers` 从4开始试；显存不足降低 batch。
- 此版本是单设备，不要直接使用 `torchrun` 多进程，会产生重复训练/输出冲突。
- 几何约定要先检查：内参随缩放变化、相机前向朝上、米/格一致、类别渲染一致。投影支持不是真实可见性；OSM未标注不能直接等同物理不存在。
- 为避免位置泄漏，按城市/不重叠地理区域划分，并人工验证地理缓冲。程序的region字符串检查不能代替地图空间审计。
- 原始 MIA 预训练是否覆盖目标评估城市需要单独核查；不要把零样本对照自动称作严格未见域结果。

已完成的实际软件检查见 [验证记录](docs/validation.md)。

## 8. 文件与可复现边界

```text
bev_osm/model.py         条件金字塔、时间U-Net、交叉注意力、边界损失
bev_osm/diffusion.py     x₀ DDIM / FM Euler / latent生成
bev_osm/codec.py         连续地图AE
bev_osm/train.py         训练、验证、断点与AE门槛
bev_osm/evaluate.py      测试、原始MIA对照、逐帧输出
bev_osm/data.py          数据格式、训练集归一化、按需加载
bev_osm/index_mia.py     官方城市目录→配对清单
bev_osm/prepare.py       配对清单→冻结BEV数据
bev_osm/export_local.py 本地KITTI实验→可迁移数据
```

整理版保留核心网络/损失/采样公式，但改成通用数据索引、均匀采样、按ID稳定噪声和完整续训机制；**不承诺重新运行即可逐位得到旧428张实验数字**。历史脚本采用按地域均衡采样、不同噪声索引和固定预算。结果表只记录既有实测值，不能当服务器版大规模训练成绩。

原始数据、缓存、权重、运行目录均被 `.gitignore` 排除；第三方代码由官方仓库独立下载。来源与许可说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。
