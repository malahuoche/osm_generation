# 来源与第三方许可

本仓库整理了本项目自己的条件生成、训练与评估代码。未将MIA/DINO/OrienterNet源码、预训练权重、数据集打包进Git，也未为第三方资产重新授权。

- MIA 官方源码：https://github.com/MapItAnywhere/MapItAnywhere ，使用固定commit `6dbfaa569c7f384204a9c4c5b915414c1c87451f`。仓库根许可证为CC BY-SA4.0；其中部分源文件另外带有源项目许可（例如OrienterNet来源文件的CC-BY-NC声明），使用时保留并遵循对应文件的声明。
- MIA权重：https://huggingface.co/mapitanywhere/mapper 。按其来源条款使用，本项目仅下载/验证并调用。
- MIA数据中的图片、地图各有对应来源条款；查看官方 https://github.com/MapItAnywhere/MapItAnywhere/blob/main/mia/dataset.md 。OSM相关数据遵循来源的ODbL要求；不要因为本仓库公开就认为原始数据可无条件重新发布。
- DiffMap参考：https://arxiv.org/abs/2405.02008 。借鉴条件地图latent扩散思路，未复制其实现，也不声明复现其完整算法。

本次整理没有替仓库所有者选择新的开源许可证；公开仓库本身不等于授予任意再许可权限。需要对本项目原创代码发布具体许可证时由所有者另行决定。

Raw-data preprocessing uses external MapMachine at commit
`2f072bba7acd6f75d7a51107fd82ba9f85cba3fc` from
https://github.com/tonyzzzzzz/map-machine and the pinned MIA rendering definitions/style.
These sources and their assets are not bundled; follow their repository licenses.
OSM XML and rendered labels derive from © OpenStreetMap contributors under ODbL;
retain attribution and review the license for redistribution of generated datasets.
The KITTI and Boreas downloads remain subject to their respective dataset terms.

The optional OrienterNet raster comparison imports the official parser, rasterizer
and palette at commit `0280b6e248046dc7d714801d7d0651a162fd5a63` from
https://github.com/facebookresearch/OrienterNet. Upstream source is not bundled;
its own license and notices apply. The displayed common palette is a diagnostic
view and does not modify the native categorical raster saved by this package.
