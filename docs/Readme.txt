使用说明及流程：
1、获得地理POI（Point of Interest）信息：
Scripts/process_poi.py中，通过Overpass API，向OpenStreetMap查询POI信息。输入为音频录制的gps位置（2个gps坐标，分别是西南方向和东北方向，代表音频录制的区域），输出为poi_text，其内容是提取区域内所有的“地理场景高级类别”：“具体类别”。比如“Amenity”: “Cinema”"。在通过API查询的过程中，使用M2M100模型将其翻译为英文，方便处理。具体的地理信息高级标签在代码中可见，共11种。结果存储在Outputs/poi_features_2.json中。

2、poi_features_2中除了包含POI信息，还包含音频事件的segment和标签（正在整理处理）。同时也包含了原始的GPS信息。

3、获得POI文本特征
Scripts/feature_POI.py，通过BERT，对于POI_text进行处理，获得文本特征

4、音频和文本模块融合：
Scripts/all_AST.py和all_CLAP.py，使用2种模型，对于音频和文本模态进行处理，使用的融合方法包括FiLM、Cross Attention、Transformer Fusion。结果保存到Output文件夹中。

5、贝叶斯统计方法：
利用地理位置信息作为先验知识，结合贝叶斯方法。
P(Class | Audio, POI) ∝ P(Class | Audio) * P(Class | POI)
P(Class | POI)：统计得到的结果，根据音频事件类别和 POI 类别的共同出现的情况计算这些概率