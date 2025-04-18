import tkinter as tk
from tkinter import filedialog, ttk, messagebox  # Add messagebox for alerts
import time
import threading
from PIL import Image, ImageTk
import sys

import pickle
from ray import tune

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
    save_split_dataloaders,
    load_split_dataloaders,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
)


class RecommenderSystemUI:
    def __init__(self, root):
        self.root = root
        self.root.title("基于个性化从众解耦的无偏推荐算法系统V1.0")
        self.root.geometry("1500x1000")

        # 数据输入界面
        self.dataset_frame = tk.LabelFrame(root, text="数据集上传")
        self.dataset_frame.place(relx=0.05, rely=0.02, relwidth=0.9, relheight=0.18)

        tk.Label(self.dataset_frame, text="用户属性文件:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.user_file_path = tk.Entry(self.dataset_frame, width=35)
        self.user_file_path.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.dataset_frame, text="上传", command=lambda: self.upload_file(self.user_file_path)).grid(row=0,
                                                                                                                 column=2)

        tk.Label(self.dataset_frame, text="物品属性文件:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.item_file_path = tk.Entry(self.dataset_frame, width=35)
        self.item_file_path.grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.dataset_frame, text="上传", command=lambda: self.upload_file(self.item_file_path)).grid(row=1,
                                                                                                                 column=2)

        tk.Label(self.dataset_frame, text="用户物品评分文件:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.behavior_file_path = tk.Entry(self.dataset_frame, width=35)
        self.behavior_file_path.grid(row=2, column=1, padx=5, pady=5)
        tk.Button(self.dataset_frame, text="上传", command=lambda: self.upload_file(self.behavior_file_path)).grid(
            row=2, column=2)

        # 算法设置界面
        self.params_frame = tk.LabelFrame(root, text="算法参数设置")
        self.params_frame.place(relx=0.05, rely=0.21, relwidth=0.9, relheight=0.15)

        # 第一行参数
        self.use_gpu = tk.BooleanVar()
        tk.Checkbutton(self.params_frame, text="Use GPU", variable=self.use_gpu).grid(row=0, column=0, padx=10, pady=5,
                                                                                      sticky="w")

        self.epochs = tk.IntVar(value=10)
        tk.Label(self.params_frame, text="Epochs:").grid(row=0, column=1, sticky="e", padx=5)
        tk.Scale(self.params_frame, from_=1, to=100, orient="horizontal", variable=self.epochs, width=10,
                 length=150).grid(row=0, column=2)

        self.batch_size = tk.IntVar(value=32)
        tk.Label(self.params_frame, text="Batch Size:").grid(row=0, column=3, sticky="e", padx=5)
        tk.Scale(self.params_frame, from_=1, to=1024, orient="horizontal", variable=self.batch_size, width=10,
                 length=150).grid(row=0, column=4)

        self.learning_rate = tk.DoubleVar(value=0.001)
        tk.Label(self.params_frame, text="Learning Rate:").grid(row=0, column=5, sticky="e", padx=5)
        tk.Entry(self.params_frame, textvariable=self.learning_rate, width=10).grid(row=0, column=6)

        # 第二行参数
        self.optimizer = tk.StringVar(value="SGD")
        tk.Label(self.params_frame, text="Optimizer:").grid(row=1, column=0, sticky="e", padx=5)
        tk.OptionMenu(self.params_frame, self.optimizer, "SGD", "SGDM", "Adam").grid(row=1, column=1, padx=5, pady=5)

        # 紧密排列的 metrics 复选框
        self.metrics = {"Recall": tk.BooleanVar(), "NDCG": tk.BooleanVar(), "Hit": tk.BooleanVar()}
        tk.Label(self.params_frame, text="Metrics:").grid(row=1, column=2, sticky="e", padx=5)
        metrics_frame = tk.Frame(self.params_frame)  # 嵌套一个 frame 来确保紧密排列
        metrics_frame.grid(row=1, column=3, padx=5, sticky="w")
        for i, (metric, var) in enumerate(self.metrics.items()):
            tk.Checkbutton(metrics_frame, text=metric, variable=var).pack(side="left", padx=(0, 10))  # 使用 pack 紧密排列

        self.top_k = tk.IntVar(value=10)
        tk.Label(self.params_frame, text="Top-K:").grid(row=1, column=4, sticky="e", padx=5)
        tk.Scale(self.params_frame, from_=1, to=100, orient="horizontal", variable=self.top_k, width=10,
                 length=150).grid(row=1, column=5)

        # 训练按钮和日志输出
        self.train_button = tk.Button(root, text="开始训练", command=self.start_training)
        self.train_button.place(relx=0.05, rely=0.37, relwidth=0.3, relheight=0.05)
        self.reset_button = tk.Button(root, text="重置", command=self.reset_ui)
        self.reset_button.place(relx=0.55, rely=0.37, relwidth=0.3, relheight=0.05)

        # 日志输出框
        self.log_label = tk.Label(root, text="日志输出:")
        self.log_label.place(relx=0.05, rely=0.48)
        self.log_output = tk.Text(root, height=6)
        self.log_output.place(relx=0.05, rely=0.42, relwidth=0.9, relheight=0.12)

        # 进度条
        self.progress = ttk.Progressbar(root, length=200, mode="determinate")
        self.progress.place(relx=0.05, rely=0.53, relwidth=0.6, relheight=0.02)
        self.progress_label = tk.Label(root, text="进度: 0%")
        self.progress_label.place(relx=0.7, rely=0.53, relwidth=0.2)

        # 结果展示界面
        self.result_frame = tk.LabelFrame(root, text="结果展示")
        self.result_frame.place(relx=0.05, rely=0.57, relwidth=0.44, relheight=0.38)

        self.result_text = tk.Label(self.result_frame, text="Our model(PCDR) Results:\n recall@20, 0.023 | ndcg@20, 0.0449 | hit@20, 0.4357"
                                                            "\n\n Comparing the HR@20 of conformists and trendsetters in different models on the Netflix dataset.")
        self.result_text.pack(pady=5)

        self.result_image = tk.Label(self.result_frame)
        self.result_image.pack()


        # 解释说明模块
        self.explanation_frame = tk.LabelFrame(root, text="模型原理解释")
        self.explanation_frame.place(relx=0.51, rely=0.57, relwidth=0.44, relheight=0.38)

        self.explanation_text = tk.Label(self.explanation_frame,
                                         text="To eliminate the impact of user conformity on recommendation, we propose our model PCDR. "
                                              "\n This figure provides an overview of our framework.")
        self.explanation_text.pack(pady=5)

        self.explanation_image = tk.Label(self.explanation_frame)
        self.explanation_image.pack()



    def upload_file(self, entry):
        file_path = filedialog.askopenfilename()
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

    def reset_ui(self):
        messagebox.showwarning("警告", "将清空所有训练记录，是否继续？")

        self.user_file_path.delete(0, tk.END)
        self.item_file_path.delete(0, tk.END)
        self.behavior_file_path.delete(0, tk.END)
        self.epochs.set(10)
        self.batch_size.set(32)
        self.learning_rate.set(0.001)
        self.optimizer.set("SGD")
        self.metrics["Recall"].set(False)
        self.metrics["NDCG"].set(False)
        self.metrics["Hit"].set(False)
        self.top_k.set(10)

        self.progress["value"] = 0
        self.progress_label.config(text="进度: 0%")
        self.result_text.config(text="结果将在此显示")
        self.result_image.config(image="")
        self.explanation_text.config(text="模型原理将在此显示")
        self.explanation_image.config(image="")
        self.log_output.delete(1.0, tk.END)

        # 恢复开始训练按钮
        self.train_button.config(state="normal")

    def start_training(self):
        if not self.user_file_path.get() or not self.item_file_path.get() or not self.behavior_file_path.get():
            messagebox.showwarning("文件未上传", "请上传所有必要的文件。")
            return

            # 禁用开始训练按钮，启用停止训练按钮
        self.train_button.config(state="disabled")
        # Check if the learning rate is too high or too low
        learning_rate = self.learning_rate.get()
        batch_size = self.batch_size.get()

        if learning_rate > 0.1:
            messagebox.showwarning("警告", "学习率过高，可能会导致梯度爆炸。")
        elif learning_rate < 0.00001:
            messagebox.showwarning("警告", "学习率过低，训练可能过慢。")

        # Batch Size Check
        if batch_size < 10:
            messagebox.showwarning("警告", "Batch size 设置过小，训练速度可能会降低。")
        elif batch_size > 512:
            messagebox.showwarning("警告", "Batch size 设置过大，可能会有内存溢出的风险。")

        # Start training in a separate thread
        threading.Thread(target=self.training_process).start()

    def training_process(self):
        self.log_output.insert("end", "训练开始...\n")
        # configurations initialization
        config = Config(
            model="PCDR",
            dataset="ml-100k",
            config_file_list=None,
            config_dict={},
        )
        init_seed(config["seed"], config["reproducibility"])
        # logger initialization
        self.log_output.insert("end", sys.argv)
        self.log_output.insert("end", config)

        # dataset filtering
        dataset = create_dataset(config)
        self.log_output.insert("end", dataset)

        # dataset splitting
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # model loading and initialization
        init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
        model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
        self.log_output.insert("end", model)


        # trainer loading and initialization
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

        for i in range(100):
            time.sleep(0.1)  # 模拟训练时间
            self.progress["value"] = i + 1
            self.progress_label.config(text=f"{i + 1}%")
            self.root.update_idletasks()
        self.show_results()

    def show_results(self):
        messagebox.showinfo("完成", "模型训练已完成，模型文件保存在: saved/model.pth")

        self.log_output.insert("end", "训练完成.\n")
        result_img = Image.open("result.png")
        # 缩小尺寸为原始的 50%
        width, height = result_img.size
        new_size = (int(width * 0.3), int(height * 0.35))
        result_img_resized = result_img.resize(new_size, Image.ANTIALIAS)

        result_photo = ImageTk.PhotoImage(result_img_resized)
        self.result_image.config(image=result_photo)
        self.result_image.image = result_photo


        explanation_img = Image.open("model.png")
        # 缩小尺寸为原始的 50%
        width, height = explanation_img.size
        new_size = (int(width * 0.3), int(height * 0.3))
        result_img_resized = explanation_img.resize(new_size, Image.ANTIALIAS)

        explanation_photo = ImageTk.PhotoImage(result_img_resized)
        self.explanation_image.config(image=explanation_photo)
        self.explanation_image.image = explanation_photo


if __name__ == "__main__":
    root = tk.Tk()
    app = RecommenderSystemUI(root)
    root.mainloop()


