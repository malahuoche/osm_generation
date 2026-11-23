# 直接处理下载的 KITTI / Boreas 原始数据

本页对应新增的原始数据适配器。以现有 MIA BEV 模型为准：0.5m/格、100×100、前方50m，车辆相机的水平前向朝上。无需下载点云。定位和标定用于构造标签；当前图像重力校正也使用已提供姿态中的重力信息，**不能称为完全不依赖姿态的纯 RGB 预处理**。严格 RGB-only 实验应另行提供重力估计并评估其误差，不能拿 GT 选最优方向。

## 0. 安装渲染依赖

先完成 README 中 Python/PyTorch、MIA 代码和 mapper 权重安装。Linux 服务器还需要：

```bash
sudo apt-get install libcairo2-dev pkg-config python3-dev
pip install -e '.[mia,geo]'
git clone https://github.com/tonyzzzzzz/map-machine.git third_party/map-machine
git -C third_party/map-machine checkout 2f072bba7acd6f75d7a51107fd82ba9f85cba3fc
export MAP_MACHINE_REPO="$PWD/third_party/map-machine"
export MIA_REPO="$PWD/third_party/MapItAnywhere"
```

渲染器调用官方 MIA 的 `draw_bev`、样式文件及其 MapMachine 分支，并按地理坐标采样到目标格点，不把地图截图直接 resize 成标签。每次渲染保存源码/样式/OSM 哈希与像素量化误差。MIA 导入只加载绘图所需的原函数定义，不启动其数据下载引擎。Linux 安装命令尚未在真实 CUDA 服务器验证；本次完整原始数据实验在 Mac CPU 完成。

## 1. KITTI：实际跑通的流程

### 1.1 所需文件

使用 **Raw 的 synced+rectified** 版本，不是 odometry 格式，也不是 unsynced/unrectified：

```text
/data/kitti/
  2011_09_26/
    calib_cam_to_cam.txt
    calib_imu_to_velo.txt
    calib_velo_to_cam.txt
    2011_09_26_drive_0005_sync/
      image_02/data/0000000000.png
      image_02/timestamps.txt
      oxts/data/0000000000.txt
      oxts/timestamps.txt
```

三个标定文件名中虽有 velo，只读取矩阵，不读取 LiDAR 测量。已下载的序列直接从1.2开始；没有数据时，可按 ZIP HTTP Range 只下载所需文件：

```bash
python -m bev_osm.download_kitti --date 2011_09_26 --drive 0005 \
  --frames 20 --out /data/kitti
```

20表示从整个序列均匀选20帧，**不是全部154帧**；`--frames 0` 提取全序列左彩色相机及配对 OXTS，仍不下载其他传感器。逐文件验证 ZIP CRC，保存 `download.json` 中的 URL、ETag、下载字节数及 SHA256。服务器不支持 Range 时会停止，不会偷偷下载整包。网络中断可重跑，现有文件须内容一致；当前下载器会重新请求选定成员，不是字节级续传。

### 1.2 从标定和位姿构造相机几何

```bash
python -m bev_osm.raw_kitti --root /data/kitti \
  --sequence 2011_09_26_drive_0005_sync --out /data/osm-generation/kitti-geometry.json
```

具体约定：

- 相机内参来自 `P_rect_02[:3,:3]`，图片尺寸必须匹配 `S_rect_02`。
- 变换链为 `T_cam2_cam0rect @ R_rect_00 @ T_cam0_velo @ T_velo_imu`。相机平移通过 `solve(K, P[:,3])` 求出，保留投影矩阵全部三维平移项。
- OXTS 的 latitude/longitude 为度；roll/pitch/yaw 为弧度；世界旋转为 `Rz(yaw) Ry(pitch) Rx(roll)`。相机中心包含 GNSS/IMU 到相机的杆臂偏移，不直接用 GPS 点作为相机位置。
- 由相机前向的水平投影定义地图朝向；求出 MIA 所需的 `Rz(roll) Rx(pitch)` 图像校正旋转，验证其使相机竖轴与重力对齐。不是直接把 OXTS roll/pitch 填给相机。
- KITTI 的原始 K 以左上像素中心为零。MIA `Camera.from_dict` 按 COLMAP 接口减0.5，所以适配器在字典中先加0.5，确保最终内部 K 恰好等于发布的 K。**只在此接口补偿一次**；官方 MIA 配对数据路径保持原规则。
- 按原始十位帧号索引两套 timestamps，记录实际时间差并拒绝超过50ms的样本。这只是守卫阈值，不代表50ms误差可忽略。

### 1.3 取得 OSM 矢量并生成标签

```bash
python -m bev_osm.fetch_osm --geometry /data/osm-generation/kitti-geometry.json \
  --out /data/osm-generation/kitti.osm
python -m bev_osm.render_raw --geometry /data/osm-generation/kitti-geometry.json \
  --osm /data/osm-generation/kitti.osm --region kitti-0005-qc --split train \
  --out /data/osm-generation/kitti-pairs
```

默认使用 OSM 官方小范围 map API，补取缺失 multipolygon 成员；也支持 `--provider overpass --endpoint https://overpass-api.de/api/interpreter`。网络限流时等待服务恢复，或通过 `--osm` 使用已下载且覆盖整个序列的 XML。大范围数据应使用本地 OSM 提取，不要对整座城市或数百万帧反复调用小范围接口。输出已存在时 fetch 拒绝覆盖，保留采集快照。

地图下载日期和影像年份不同，生成的是 **OSM 配对弱标签**。道路宽度沿用 MIA 样式，并非实测宽度；OSM 缺失也不代表现实不存在。渲染前检查 way 引用和 multipolygon 成员完整性；但这种检查不能证明 OSM 本身标注完整。

### 1.4 提取 BEV 并输出可视化检查

```bash
python -m bev_osm.prepare --pairs /data/osm-generation/kitti-pairs/pairs.jsonl \
  --out /data/osm-generation/kitti-qc --mia-checkpoint weights/mapper-excl-ood.ckpt \
  --device cuda --qc-only
python -m bev_osm.raw_report --geometry /data/osm-generation/kitti-geometry.json \
  --prepared /data/osm-generation/kitti-qc --out /data/osm-generation/kitti-report
```

打开 `kitti-report/index.html`，逐帧比较原图、实际校正输入、完整标签和投影范围内标签。`audit.json` 保存格式/哈希、主点换算和姿态误差检查。深色是支持区外；白色表示六层没有标注，不能一概当成自由空间。

`--qc-only` 允许单序列检查，**不是训练/验证/测试数据集**。其 manifest 默认不能用于训练，因为缺少独立的val/test。正式训练时，对地理不重叠的区域分别运行 raw_kitti/render_raw，设置各自 region 和 split，再合并 pairs.jsonl，运行 prepare（不带qc-only）。合并前检查不同 drive 是否经过同一地点；仅序列名不同不足以排除泄漏。不要复制样本来填满三个 split。

## 2. Boreas：输入约定与适配命令

官方图片已经 rectified，不再次使用 `camera0_intrinsics.yaml` 去畸变。所需文件：

```text
boreas-YYYY-MM-DD-HH-MM/
  camera/<整数微秒时间戳>.png
  calib/P_camera.txt
  applanix/camera_poses.csv
  applanix/gps_post_process.csv
```

此外需要参考序列 `boreas-2020-11-26-13-58` 的 GPS 首行，确定所有序列共用 ENU 的方向。可下载这份 GPS 文件作为 `--reference-gps`；不需要下载参考序列全部图片。

下载自己需要的序列元数据示例（AWS CLI）：

```bash
aws s3 cp s3://boreas/boreas-2020-11-26-13-58/calib/P_camera.txt /data/boreas/boreas-2020-11-26-13-58/calib/P_camera.txt --no-sign-request
aws s3 cp s3://boreas/boreas-2020-11-26-13-58/applanix/camera_poses.csv /data/boreas/boreas-2020-11-26-13-58/applanix/camera_poses.csv --no-sign-request
aws s3 cp s3://boreas/boreas-2020-11-26-13-58/applanix/gps_post_process.csv /data/boreas/boreas-2020-11-26-13-58/applanix/gps_post_process.csv --no-sign-request
```

图片按 camera_poses 第一列时间戳逐张 `aws s3 cp`；已有图片时无需重下。不要 `sync` 全序列所有传感器。

```bash
python -m bev_osm.raw_boreas \
  --sequence-dir /data/boreas/boreas-2020-11-26-13-58 \
  --gps /data/boreas/boreas-2020-11-26-13-58/applanix/gps_post_process.csv \
  --reference-gps /data/boreas/boreas-2020-11-26-13-58/applanix/gps_post_process.csv \
  --out /data/osm-generation/boreas-geometry.json
```

后续同KITTI，向 fetch_osm/render_raw/raw_report 传入 boreas-geometry.json。首次检查可加 `--limit 1`。当前适配器针对原版 Boreas，不承诺兼容 Boreas-RT。

已核对的重要差异：

- `P_camera.txt` 实际可为4×4；使用其上方3×4并检查最后一行为 `[0,0,0,1]`。当前仅接受中心投影平移列为零。
- `camera_poses.csv` 已是**相机**在共用 ENU 的位姿，不再重复乘相机外参。
- 相机时间戳为微秒整数；`gps_post_process.csv` 的时间为秒，latitude/longitude为弧度。GPS easting/northing是相对 ENU 米坐标，不是经纬度或直接可当 UTM 的数值。
- 旋转严格对应官方 `yawPitchRollToRot` 的被动旋转约定，而非套用 KITTI 欧拉角公式。
- 相机时间必须在GPS记录内插区间，要求相邻GPS间隔≤20ms，不进行越界外推。相机与 Applanix 位置之差用于杆臂偏移，并把参考ENU方向转换到当前地理点的局部ENU。
- 地图采用高度零的局部地理基准，**不是实际地面高程**；目前没有地形坡度恢复。相机/地图配准仍需视觉检查。

## 3. 本次实测范围（2026-11-23）

- KITTI `2011_09_26_drive_0005_sync`：从官方S3新下载20帧及定位/标定，实际传输约17.285MB；完整ZIP约645.94MB。
- 20帧全部完成原版MIA渲染和冻结BEV提取。最大相机/OXTS时间差11.765ms；已保存逐帧数值和可视化，不宣称具有独立测绘精度。
- 测试覆盖投影平移完整公式、重力校正、主点接口半像素换算、缓存形状和哈希。
- Boreas：利用本地已有首帧和新下载的首段GPS核对，旋转与官方源码数值一致；实际4×4投影、时间单位及地理坐标转换已运行。OSM下载限流，**本次没有完成Boreas标签→BEV的端到端实测**，不能把KITTI通过当作Boreas也通过。
- 单序列QC，不训练新模型、不报告泛化指标；没有使用LiDAR点云。

## 官方核对来源

- [KITTI Raw：同步和校正版本](https://www.cvlibs.net/datasets/kitti/raw_data.php)
- [KITTI网站链接的pykitti标定加载实现](https://github.com/utiasSTARS/pykitti/blob/master/pykitti/raw.py)：本实现保留P的完整平移项，额外做直接投影等价性检查。
- [Boreas数据说明](https://github.com/utiasASRL/pyboreas/blob/master/DATA_REFERENCE.md)
- [Boreas旋转和位姿实现](https://github.com/utiasASRL/pyboreas/blob/master/pyboreas/utils/utils.py)
- [Boreas标定读取实现](https://github.com/utiasASRL/pyboreas/blob/master/pyboreas/data/calib.py)
- [MIA原始渲染器](https://github.com/MapItAnywhere/MapItAnywhere/blob/6dbfaa569c7f384204a9c4c5b915414c1c87451f/mia/bev/get_bev.py)

以上说明与本次实际下载文件共同核对；不能只用默认参数替代每个序列真实的标定。
