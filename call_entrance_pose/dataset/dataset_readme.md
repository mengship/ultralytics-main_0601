json为标注文件
"label": "oil"为标记了整个油表框
"label": "center",为中心点坐标
"label": "tip",为指针坐标
"label": "empty",为起始位置坐标
"label": "full",为最大位置坐标

center 到 empty 组成起始线
center 到 tip 组成指针线
center 到 full 组成 最大线

从起始线到最大线有两个方向，取角度较大的方向为正确方向
沿着该方向，从起始线到指针线为指针角度
沿着该方向，从起始线到最大线为最大角度

指针角度 / 最大角度 则为当前油表比例
