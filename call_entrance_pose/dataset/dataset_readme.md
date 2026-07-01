json为标注文件
"label": "oil"为标记了整个油表框
"label": "center",为中心点坐标
"label": "tip",为指针坐标
"label": "empty",为起始位置坐标（空油线）
"label": "full",为最大位置坐标（满油线）

center 到 empty 组成空油起始线
center 到 tip 组成当前指针线
center 到 full 组成满油线

角度计算规则：
1. 从 empty 到 full 和从 empty 到 tip 分别计算顺时针和逆时针两个方向的角度
2. 选择能让 tip 位于 empty 和 full 之间的方向（即 empty→tip 角度 < empty→full 角度）
3. 如果两个方向都满足条件，选择 empty→full 角度更大的方向
4. 如果两个方向都不满足，则回退到选择 empty→full 角度更大的方向
5. 在选定方向下：指针角度 / 最大角度 = 当前油表比例（fuel_ratio）
6. 如果比例超出 [0, 1] 范围，则截断到该范围内，但同时保留 raw_fuel_ratio 用于调试
