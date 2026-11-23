# OrienterNet raster 对照组

此对照比较**标签表示与栅格规则**，不是新训练的模型成绩。Boreas首帧和KITTI的20帧均已运行。

|项目|MIA标签|OrienterNet原生raster|
|---|---|---|
|数据组织|六个二值层|三个类别编号层：areas、ways、nodes|
|建筑|填充区域|填充区域，另有 building_outline 线类别|
|普通道路|按MIA样式渲染宽度|原生1像素线；显式标注的道路面仍可按面处理|
|背景|六层均未标注|各层ID=0；不等于已确认自由空间|
|重叠|各二值层可重叠|同一类别层按原生类别列表优先级选一个ID；层间可重叠|
|默认配色|本工程MIA六类颜色|官方Colormap，默认只合成areas和ways；nodes另行展示|

## 如何保证是同一范围

- 两组必须使用同一份OSM XML：脚本比对MIA渲染审计记录中的SHA256，不相同则拒绝运行。
- 使用同一相机位置、水平朝向、100×100格点、0.5m/格，以及相同图像投影支持掩膜。
- OrienterNet执行官方 `OSMData.from_file → MapData.from_osm → render_raster_masks → render_raster_map`。没有手写一套类似样式替代它。
- 在调用原版栅格函数前，将OSM节点转到相机的right/forward坐标。保持类别、绘图宽度和优先级不变，避免旋转已生成的一像素线图带来第二次量化。**这属于原生规则的车辆朝向适配，不是作者north-up缓存截图。**
- Canvas边界为 `[-25.25,0.25]` 到 `[24.75,50.25]`，ppm=2，严格对应像素中心 `right=0.5c−25`、`forward=50−0.5r`。相机连续坐标为(row=100,col=50)。
- 地理格点转换对照：Boreas最大差异小于0.000001m；KITTI20帧最大差异约0.00102m。这里衡量的是代码坐标转换的一致性，不是GPS、OSM或栅格边界的绝对测绘精度。

## 服务器上运行

已有原始数据预处理输出后，安装轻量渲染依赖并取得官方源码：

```bash
pip install -e '.[geo,orienternet]'
git clone https://github.com/facebookresearch/OrienterNet.git third_party/OrienterNet
git -C third_party/OrienterNet checkout 0280b6e248046dc7d714801d7d0651a162fd5a63
export ORIENTERNET_REPO="$PWD/third_party/OrienterNet"

python -m bev_osm.compare_rasters \
  --geometry /data/osm-generation/boreas-geometry.json \
  --osm /data/osm-generation/boreas.osm \
  --prepared /data/osm-generation/boreas-qc \
  --mia-audit /data/osm-generation/boreas-pairs/render-audit.json \
  --out /data/osm-generation/boreas-report
```

路径必须指向之前同一批预处理结果。KITTI用相应的geometry、OSM、prepared和render-audit文件替换即可。不需要OrienterNet定位权重，也不运行定位网络。

输出：

- `index.html`：原图、MIA、OrienterNet官方配色、OrienterNet统一配色；上下两排分别为完整范围与同一支持区。
- `orienternet_*.npz`：原生 `raster[3,100,100]` 整数类别ID与支持掩膜。统一配色不会改动数组。
- `orienternet-audit.json`：完整类别顺序、源码及OSM哈希、坐标约定。必须连同NPZ保存，否则无法解释ID。
- `orienternet_*_nodes.png`：用黑点单独叠加点要素，避免官方默认配色不显示nodes造成误解。
- 单帧报告另提供前方120m、左右75m的上下文图，用蓝框标出实际50m裁剪范围。
- 若输出目录已有预处理页面，首次保存为 `raw-qc.html`；原图文件不覆盖。

## 本次观察与纠正

Boreas首帧，两组左侧建筑形状接近，道路由MIA的宽面变为OrienterNet的细线。不能用两者道路面积IoU判断哪个更准确，因为表示定义不同。

此前关于右侧圆顶“未在六类标签中体现，可能标注/渲染缺失”的解释不充分。扩大范围后，OSM way 66984065 是附近一栋圆形建筑，很可能对应可见圆顶；其轮廓约在相机右方9.5–61.6m、前方46.4–107.1m，主体落在50m裁剪之外。应优先解释为范围限制，不能据此认定OSM缺失。仅凭图像仍不能保证物体身份完全对应。

当前六层二值生成模型不能直接把这三个类别ID层当作训练目标。若后续训练此对照，应为三层分别设置类别预测头/损失，或显式设计one-hot表示并记录映射；不能把类别编号当连续灰度回归。

## 来源与核验

- [官方栅格函数](https://github.com/facebookresearch/OrienterNet/blob/0280b6e248046dc7d714801d7d0651a162fd5a63/maploc/osm/raster.py)
- [官方类别规则](https://github.com/facebookresearch/OrienterNet/blob/0280b6e248046dc7d714801d7d0651a162fd5a63/maploc/osm/parser.py)
- [官方配色](https://github.com/facebookresearch/OrienterNet/blob/0280b6e248046dc7d714801d7d0651a162fd5a63/maploc/osm/viz.py)

本次核验7个实际调用的源码文件与官方固定提交内容逐字节一致；本地包含原生栅格检查在内的10项测试通过。未进行新一轮模型训练。
