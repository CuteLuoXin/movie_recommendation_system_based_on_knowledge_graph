import tensorflow as tf
from typing import List
from KGCN.layer import SumAggregator

# 定义自定义对象字典
custom_objects = {'SumAggregator': SumAggregator}

# 加载已保存的模型，并传递自定义对象字典
model_path = "model/KGCN.h5"
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# 准备新数据用于预测
user_id = 0  # 用户ID
item_id = 0  # 物品ID

# 将新数据转换为适合输入模型的形式
user_id_input = tf.convert_to_tensor([user_id])
item_id_input = tf.convert_to_tensor([item_id])

# 进行预测
prediction = model.predict([user_id_input, item_id_input])

# 打印预测结果
print("预测结果:", prediction)
