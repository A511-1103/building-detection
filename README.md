# building-detection
semantic segmentation,DeepLab,U-Net,SCSENet,BAM,HRNet,DenseASPP,Res-UNet,Remote sensing,tensorflow2  
  
# 项目介绍  
  遥感影像建筑物检测任务  
目的：简化工作人员的建筑物轮廓目视解析任务   
方法：使用卷积神经网络，在WHU建筑物数据集上进行训练，后使用自己的数据做迁移学习，
  通过对5个模型预测结果的融合，得到最终的检测结果。  
提供了两种预测方法：  
1、在本地执行predict.py，对输入影像做建筑物检测  
2、使用flask，把检测代码封装成API，给外部提供访问接口，请求方式为POST  
  
# 文件介绍  
1、CLient    :   存放的是远程访问的客户端；  
2、predict_model :   5个检测模块  
3、receive_file  :   存放每个客户端传来的数据(影像)  
4、train_model   :   5个神金网络模型  
5、buildAPI.py   :   提供外部访问的接口  
6、edge_3.py     :   轮廓优化代码  
7、model_fuse.py :   模型融合代码  
8、predict.py    :   整理后的预测代码  
9、data_enhancement.py ： 数据增强模块+划分训练集与验证集  

  
# 运行方式  
方法一：在服务端运行buildAPI.py,客户端将影像以二进制传入，
以JSON形式返回，结果为base64格式的检测结果图，以及优化后的轮廓点  
方法二：运行 8、predict.py，提前修改好待检测影像路径  
  
# 作者  
长沙理工大学1103实验室  
  
  
# 待续...

