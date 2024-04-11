import gradio as gr
import tkinter as tk
import threading
import tensorflow as tf
from KGCN.layer import SumAggregator
custom_objects = {'SumAggregator': SumAggregator}


def get_top_five_name(item_id):
    file_name_list= []
    with open("../data/ds/ml-1m/movies.dat", "r") as file:
        for line in file:
            data = line.strip().split("::")
            if len(data) >= 2 and data[0] == item_id:
                second_column_data = data[1]
                file_name_list.append(second_column_data)
    return file_name_list

def read_data(user_id):
    seccon_list= []
    with open("../data/ds/ml-1m/ratings.dat", "r") as file:
        for line in file:
            data = line.strip().split("::")
            if len(data) >= 2 and data[0] == user_id:
                second_column_data = data[1]
                seccon_list.append(second_column_data)
    return seccon_list
def recommend_movies(user_id):
    model_path = "../model/KGCN.h5"
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
    data=list(zip(predection_list_predection,predection_list_item))
    sorted_data = sorted(data, key=lambda x: x[0], reverse=True)[:18]
    sorted_array1, sorted_array2 = zip(*sorted_data)
    message = ""
    for i in range(len(sorted_array1)):
        message += f"推荐：{sorted_array2[i]}, 喜欢可能性：{sorted_array1[i]}\n"
    return message

with gr.Blocks() as demo:
    gr.Markdown("电影推荐系统")
    with gr.Row():
        inp = gr.Textbox(placeholder="输入用户ID", label="输入ID")
        out = gr.Textbox(lines=10, label="推荐电影列表")  # Add label for the output textbox
    btn = gr.Button("预测")
    btn.click(fn=recommend_movies, inputs=inp, outputs=out)

demo.launch()
