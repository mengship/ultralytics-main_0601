json为标注文件
"label": "oil"为标记了整个油表框
"label": "center",为中心点坐标
"label": "tip",为指针坐标
"label": "empty",为起始位置坐标
"label": "full",为最大位置坐标

center 到 empty 组成起始线
center 到 tip 组成指针线
center 到 full 组成最大线

角度计算规则：
1. 从 empty 到 full 有顺时针和逆时针两个方向
2. 计算这两个方向的角度，取角度更大的方向作为有效方向
3. 沿着有效方向，从 empty 到 tip 的角度为指针角度
4. 沿着有效方向，从 empty 到 full 的角度为最大角度（max_angle）
5. 指针角度 / 最大角度 = 当前油表比例（fuel_ratio）
6. 如果比例超出 [0, 1] 范围，则截断到该范围内，但同时保留 raw_fuel_ratio 用于调试
