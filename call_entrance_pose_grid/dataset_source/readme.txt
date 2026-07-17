数据路径
'/Users/flash/Documents/Data_Work/99_临时中转站/9 潘杰/数据标记/test'
其中 readme.xlsx 是说明文件，里面的sheet1有三列： 图片名称	油表类型	油表位置
油表类型：格子/指针
油表位置：lower_right/top_right/left/lower_left
除了readme.xlsx其他都是图片文件和标注文件
有三个需求：
1、按照油表类型的 格子 和 指针，按照8：2生成yolo检测格式数据集，生成对应的数据处理脚本
2、针对油表类型是 格子的，按照8：2生成yolo pose检测格式数据集，生成对应的数据处理脚本 ， oil1 / empty / full / tip
3、针对油表类型是 指针的，按照油表位置类型按照8：2生成yolo pose检测格式数据集，生成对应的数据处理脚本 ，oil / center / empty / full / tip