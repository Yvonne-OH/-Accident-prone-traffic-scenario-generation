# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 19:48:48 2024

@author: 39829
"""


import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import lanelet2
from lanelet2.core import Lanelet, LineString3d, Point3d, BasicPoint3d, BasicPolygon3d, RegulatoryElement
from lanelet2.extension import createRegulatoryElement, LaneletPath, generateGuidanceLines, getCenterline

# 假设你有一个 Lanelet 地图和 OSM 地图
# 在实际情况中，你需要加载相关的地图数据
lanelet_map = lanelet2.io.load("C:\\Users\\39829\\Desktop\\SocialVAE\\data\\INTERACTION\\INTERACTION-Dataset-DR-multi-v1_2\\maps\\DR_CHN_Merging_ZS0.osm")

# 创建一个图形
fig, ax = plt.subplots()

# 获取所有 Lanelet
all_lanelets = lanelet_map.laneletLayer.lanelets

# 绘制每个 Lanelet
for lanelet in all_lanelets:
    polygon = Polygon([p.x, p.y] for p in lanelet.polygon3d().basicPolygon().vertices())
    ax.add_patch(polygon)

    # 绘制中心线
    centerline = getCenterline(lanelet).basicLineString()
    ax.plot(centerline.x(), centerline.y(), color='black')

    # 绘制关联的 RegulatoryElement
    for reg_elem in lanelet.regulatoryElements:
        if isinstance(reg_elem, RegulatoryElement):
            # 在这里添加逻辑以根据 RegulatoryElement 的类型绘制相应的信息
            # 示例：绘制禁停标志
            if reg_elem.type == lanelet2.regulatory.LaneletMarking.NoParking:
                no_parking_area = reg_elem.polygon3d().basicPolygon().toPolygon2d()
                no_parking_polygon = Polygon([p.x, p.y] for p in no_parking_area.vertices())
                ax.add_patch(no_parking_polygon)
                
# 设置坐标轴
ax.set_aspect('equal', 'box')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图形
plt.show()
