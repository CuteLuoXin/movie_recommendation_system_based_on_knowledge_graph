import tkinter as tk
import threading
import tensorflow as tf
from KGCN.layer import SumAggregator
custom_objects = {'SumAggregator': SumAggregator}


def get_top_five_name(item_id):
    file_name_list= []
    with open("data/ds/ml-1m/movies.dat", "r") as file:
        for line in file:
            data = line.strip().split("::")
            if len(data) >= 2 and data[0] == item_id:
                second_column_data = data[1]
                file_name_list.append(second_column_data)
    return file_name_list

def read_data(user_id):
    seccon_list= []
    with open("data/ds/ml-1m/ratings.dat", "r") as file:
        for line in file:
            data = line.strip().split("::")
            if len(data) >= 2 and data[0] == user_id:
                second_column_data = data[1]
                seccon_list.append(second_column_data)
    return seccon_list
def recommend_movies(user_id):
    model_path = "model/KGCN.h5"
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    predection_list_predection = []
    predection_list_item = []
    item_ids = read_data(user_id)
    for item_id in item_ids:
        user_id_input = tf.convert_to_tensor([int(user_id)])
        item_id_input = tf.convert_to_tensor([int(item_id)])
        prediction = model.predict([user_id_input, item_id_input])
        # 通过预测用户对某一个新的电影的可能喜欢成都来为其推荐电影。
        predection_list_predection.append(prediction)
        predection_list_item.append(get_top_five_name(item_id))

    return predection_list_predection,predection_list_item
def on_button_click(entry_widget, text_widget):
    message_clear = f""
    text_widget.insert(tk.END, message_clear)
    message = f"正在加载模型，请等待...\n"
    text_widget.insert(tk.END, message)
    user_id = entry_widget.get()
    print(user_id)
    sorted_tuple_list_prediction,sorted_tuple_list_item = recommend_movies(user_id,text_widget)
    data=list(zip(sorted_tuple_list_prediction,sorted_tuple_list_item))
    sorted_data = sorted(data, key=lambda x: x[0], reverse=True)[:18]
    sorted_array1, sorted_array2 = zip(*sorted_data)
    print(sorted_array1)
    for i in range(len(sorted_array1)):
        message1 = f"推荐：{sorted_array2[i]},喜欢可能性：{sorted_array1[i]}\n"
        text_widget.insert(tk.END, message1)

def run_gui():
    # 创建主窗口
    root = tk.Tk()
    root.title("电影推荐系统")

    # 创建标签和输入框
    label1 = tk.Label(root, text="用户编号:")
    label1.grid(row=0, column=0, padx=5, pady=5)

    entry1 = tk.Entry(root)
    entry1.grid(row=0, column=1, padx=5, pady=5)

    # 创建消息框
    text = tk.Text(root, width=70)  # 设置消息框宽度为30
    text.grid(row=2, columnspan=2, padx=5, pady=5)


    # 创建按钮
    button = tk.Button(root, text="开始推荐", command=lambda: on_button_click(entry1, text))
    button.grid(row=3, columnspan=2, padx=5, pady=5)

    # 运行界面
    root.mainloop()

# 在新线程中启动UI界面
ui_thread = threading.Thread(target=run_gui)
ui_thread.start()
