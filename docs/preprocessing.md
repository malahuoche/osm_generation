# 数据预处理：从下载文件到可训练的 manifest

所有命令在本仓库根目录、已安装 README 第1节依赖的 Python 环境执行。当前实现冻结 MIA：**RGB → 缓存 BEV → 训练生成模型**，训练阶段不再读取 RGB。缓存不能用于端到端解冻 MIA。

原始KITTI/Boreas的新增入口与实测范围见 [原始驾驶数据流程](raw-driving-data.md)。

## 先选对入口

|你手里的数据|入口|是否还需要生成 GT|
|---|---|---|
|本次实验的 KITTI 428 张配对标签及特征|下面 A：export_local|不需要|
|官方 MIA 配对数据，含图片、semantic_masks、dump.json|下面 B：index_mia → prepare|不需要|
|只有 KITTI/Boreas 原始图片、位姿与标定|下面 C|需要，使用新增 raw_* → fetch_osm → render_raw 流程|

## A. 搬迁已有的 428 张实验数据

在原来的 Mac 上设置真实实验根目录（不是 Git 仓库目录）：

```bash
export EXPERIMENTS_ROOT=/你的实验目录/experiments
python -m bev_osm.export_local \
  --source "$EXPERIMENTS_ROOT/mia-v3-locked/data/kitti-native-mia-428" \
  --features "$EXPERIMENTS_ROOT/mia-minimum/data/latent-osm/raw_features128.npy" \
  --out ./data/kitti428
python -m bev_osm.audit --manifest ./data/kitti428/manifest.jsonl --verify-hashes
```

`--source` 内必须有 `metadata.json` 和 `targets.npz`。导出保留原实验的训练/验证/测试划分与评估掩膜，不重新提取特征，不需要 GPU 或 MIA 权重。预期428张：train210 / val125 / test93；以 audit 实际输出为准。

把 **整个 kitti428 目录** 传到服务器，不能只传 manifest：

```bash
rsync -av --progress ./data/kitti428/ USER@SERVER:/data/osm-generation/kitti428/
```

将 USER、SERVER 替换为服务器账号和地址。在服务器再次运行 audit，然后设置：

```bash
export OSM_MANIFEST=/data/osm-generation/kitti428/manifest.jsonl
```

即可执行 README 训练命令。这是已有小规模实验数据的迁移，不是扩大数据集。

## B. 处理官方 MIA 配对数据

### 1. 解压并检查目录

下载地址、MIA 代码及权重安装见 README 第2、3节。`--root` 指向城市文件夹的共同父目录，要求如下（城市名、图片 ID 仅为示意）：

```text
/data/osm-generation/mia_raw/
  city_a/
    dump.json
    images/
      123_undistorted.jpg
    semantic_masks/
      123.npz
    flood_fill/                 # 可选；默认不用
      123.npz
  city_b/ ...
  city_c/ ...
```

实际图片文件名来自 dump 中的 view key，可为 jpg/png/jpeg；地图文件名为去掉 `_undistorted` 后的 ID。dump 的每个序列需包含 `views` 和 `cameras`；view 含 `camera_id`、`roll_pitch_yaw`，camera 必须为已去畸变的 `PINHOLE`。不要把原始畸变图重命名后冒充去畸变图。

### 2. 先按地域划分，再提取特征

新建 `configs/my_city_splits.json`，填写**实际下载的目录名**：

```json
{"city_a":"train","city_b":"val","city_c":"test"}
```

三个集合均须非空。不要将同一区域的相邻帧随机分到不同集合，也不要把同一城市复制成三个名字绕过检查。程序只能检查 region 字符串不同；实际地理范围是否重叠需要依据坐标另行检查。样本少到无法留出独立地域时，只做流程测试，不能声称泛化评测。

```bash
python -m bev_osm.index_mia \
  --root /data/osm-generation/mia_raw \
  --splits configs/my_city_splits.json \
  --out /data/osm-generation/pairs.jsonl
cat /data/osm-generation/pairs.audit.json
```

检查 `indexed` 与 `missing_image_or_map`。缺图或地图的记录会跳过；数量不符合预期时先补下载，不要直接训练。索引已存在时需换输出名，避免误覆盖。

### 3. 提取冻结的 MIA 特征

```bash
export MIA_REPO="$PWD/third_party/MapItAnywhere"
python -m bev_osm.prepare \
  --pairs /data/osm-generation/pairs.jsonl \
  --out /data/osm-generation/prepared \
  --mia-checkpoint weights/mapper-excl-ood.ckpt \
  --device cuda
```

代码实际进行以下操作：

1. 读取 RGB，检查尺寸与相机标定一致，转换到 `[0,1]`；按 roll/pitch（度）重力校正。
2. 最长边缩放至512，补齐512方形，随操作更新相机内参；**不要提前手动拉伸图片或独立修改内参**。
3. MIA 图像编码器 → scale classifier → polar projection → BEV projection；全程冻结，不使用 mapper 预测作为条件。
4. 裁去 BEV 最后一列，保存 `128×100×100` 的 float16 特征与投影支持掩膜。
5. 从官方地图中心取前方100行、左右各50列；选择原通道 `[0,1,2,4,6,7]`，得到6层二值 GT。

GT 是已有标签的裁剪，**这一步不下载 OSM、不估计朝向，也不把任意地图自动旋转对齐**。输入原生地图必须已满足官方坐标与尺度约定。最终范围为前方50m、左右各25m，0.5m/格；车前向上，相机位于底边中心连续坐标 `(row=100,col=50)`，不是有效数组下标。

六层依次为 road、crossing、sidewalk、building、terrain、parking；允许层间重叠，不是彩色PNG，也不是单通道类别编号。白色渲染区域不能自动解释为确定的自由空间。

默认 `eval_mask=support`。support 仅是投影可支持区域，**不是实际遮挡可见性**。若每条记录都有 flood 文件，可选择 `--flood-eval-mask`，将评估/损失范围进一步限制；该 GT 派生掩膜不输入生成模型。新旧实验必须采用相同评估范围才能直接比较。

中断后重跑同一命令并追加 `--resume`。不要在续提取期间修改输入图片/地图；更换清单、权重或掩膜策略需新输出目录。代码对清单和权重做签名，但不会逐一检测续跑前原始图片是否被替换。

### 4. 检查输出，再启动训练

```text
prepared/
  prepare.json
  manifest.jsonl
  manifest.meta.json
  samples/
    <样本ID的哈希>.npz
```

每个 NPZ 含 `bev[128,100,100]`、`target[6,100,100]`、`support[1,100,100]`、`eval_mask[1,100,100]`。manifest 用相对路径，可整体搬迁。每帧仅 BEV 未压缩约2.56MB；先估算磁盘空间。

```bash
python -m bev_osm.audit \
  --manifest /data/osm-generation/prepared/manifest.jsonl --verify-hashes
export OSM_MANIFEST=/data/osm-generation/prepared/manifest.jsonl
```

看到 `status: passed` 仅说明格式、区域名称隔离、文件哈希通过，不代表几何对齐正确。训练前抽查各集合的 RGB 与 GT：前方是否朝上、路口左右是否对应、道路是否从相机附近向前延伸、尺度是否合理。不要通过随机旋转、缩放 GT 来“看起来对齐”。特征归一化由训练程序仅用 train 计算并存入检查点，无需自己对整个数据集计算均值方差。

## C. 新 KITTI / Boreas 或其他自建配对数据

**原始驾驶数据不能直接运行 index_mia。请先按 [新增原始数据流程](raw-driving-data.md) 构造配对标签，再提取BEV。** 需要先用相机位姿、相机/车体坐标变换及对应 OSM 提取，在一致尺度、朝向、渲染规则下生成标签；道路宽度渲染假设、地图时间差和缺失标注需记录。不能用标签反向挑选最优相机方向。无需 LiDAR。

准备好后可按 `configs/pair.example.jsonl` 为每帧写一行，包含唯一 id、split、region、image、target、target_format、camera、roll_pitch_degrees。路径相对 pairs.jsonl 所在目录；camera.params 为 PINHOLE 的 `[fx,fy,cx,cy]`，width/height 对应输入图片。不要照抄示例内参，也不要把未知 roll/pitch 一律当作零。

`target_format=six_chw`：单数组 NPY/NPZ，形状6×100×100、值0/1，严格满足上述类别、尺度和前向约定；不再裁剪。`mia_native`：官方 HWC≥8 层格式，按步骤 B 裁剪。之后直接运行 prepare 和 audit。当前代码固定100×100，换尺度或范围不是简单 resize，需要同步修改投影、网络和评估约定。

## 常见错误定位

|现象|处理|
|找不到 dump.json|root 应指向城市目录的父目录，检查解压是否多套一层|
|Require nonempty train/val/test|检查划分和缺失文件计数，不要复制数据充数|
|Image dimensions disagree|用与图片匹配的标定，避免事先缩图但未改内参|
|Expected one array|输入地图 NPZ 只能存一张数组；不能直接把多字段缓存当原始GT|
|Region overlap / Duplicate id|修复真实重复或地理划分，不要只改名字|
|OOM|prepare 当前逐帧提取，检查设备其他进程；CPU可用 `--device cpu`，速度较慢|
