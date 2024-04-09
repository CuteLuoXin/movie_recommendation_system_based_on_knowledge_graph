import tkinter as tk

def on_button_click():
    input_text1 = entry1.get()
    input_text2 = entry2.get()
    message = f"输入框1内容：{input_text1}\n输入框2内容：{input_text2}"
    text.insert(tk.END, message)

# 创建主窗口
root = tk.Tk()
root.title("左右分布的界面示例")

# 创建标签和输入框
label1 = tk.Label(root, text="用户编号:")
label1.grid(row=0, column=0, padx=5, pady=5)

entry1 = tk.Entry(root)
entry1.grid(row=0, column=1, padx=5, pady=5)

label2 = tk.Label(root, text="电影编号:")
label2.grid(row=1, column=0, padx=5, pady=5)

entry2 = tk.Entry(root)
entry2.grid(row=1, column=1, padx=5, pady=5)

# 创建消息框
text = tk.Text(root, width=30)  # 设置消息框宽度为30
text.grid(row=2, columnspan=2, padx=5, pady=5)

# 创建按钮
button = tk.Button(root, text="点击这里", command=on_button_click)
button.grid(row=3, columnspan=2, padx=5, pady=5)

# 运行界面
root.mainloop()
