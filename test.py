import tensorflow as tf
from typing import List
from KGCN.layer import SumAggregator

# 定义自定义对象字典
custom_objects = {'SumAggregator': SumAggregator}

# 加载已保存的模型，并传递自定义对象字典
model_path = "model/KGCN.h5"
print("加载模型...")
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
print("加载完毕！")
while True:
    # 输入用户ID和物品ID
    user_id = int(input("请输入用户ID: "))
    item_id = int(input("请输入电影ID: "))

    # 将新数据转换为适合输入模型的形式
    user_id_input = tf.convert_to_tensor([user_id])
    item_id_input = tf.convert_to_tensor([item_id])

    print("预测中...")
    # 进行预测
    prediction = model.predict([user_id_input, item_id_input])

    # 通过预测用户对某一个新的电影的可能喜欢成都来为其推荐电影。

    print("用户喜欢电影{}的概率为:{}".format(item_id, prediction))
