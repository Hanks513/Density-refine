import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
from PIL import Image
import cv2
import time
import multiprocessing
import argparse
from functools import partial

rootpath = "./patent_data/"

def write_to_file(queue, file_path):
    with open(file_path, 'a') as file:
        while True:
            data = queue.get()
            if data is None:
                break
            file.write(data)

def New_data_maker(line, save_rootpath, txt_filepath, db_eps, queue):
    img_path = line[0]
    label = line[1]
    if int(label) % 100 == 0:
        print(f"label: {label}")
    directory, filename = os.path.split(img_path)
    # with open(txt_filepath, 'a') as file:
    #     file.write(f"{img_path} {label}\n")
    queue.put(f"{img_path} {label}\n")
    img = cv2.imread(rootpath+img_path)
    # print(rootpath+img_path)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # print(img)
    nonzero_points = np.column_stack(np.where(img > 0))
    sample_size = 400000
    if len(nonzero_points) > sample_size:
        nonzero_points = nonzero_points[np.random.choice(nonzero_points.shape[0], sample_size, replace=False)]
    
    dbscan = DBSCAN(eps=db_eps, min_samples=2)
    labels = dbscan.fit_predict(nonzero_points)
    unique_labels = np.unique(labels[labels != -1])

    # 繪製旋轉後的分群結果
    nonzero_points_xy = np.column_stack((nonzero_points[:, 1], nonzero_points[:, 0]))
    coordinates = nonzero_points_xy[:, :2]

    BBox = []
    for cluster_label in unique_labels:
        # 提取属于当前簇的点坐标
        cluster_points = coordinates[labels == cluster_label]
        
        if len(cluster_points) > 30000:
            # 计算 bounding box
            min_x, min_y = np.min(cluster_points, axis=0)
            max_x, max_y = np.max(cluster_points, axis=0)

            # 绘制 bounding box
            rectangle = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                    edgecolor='red', linewidth=2, fill=False)
            xy = [rectangle.get_x(),rectangle.get_y()]
            width = rectangle.get_width()
            height = rectangle.get_height()

            # 計算Rectangle的四個角點
            left = (xy[0])
            upper = (xy[1])
            right = (xy[0] + width)
            lower = (xy[1] + height)

            # 建立Rectangle對應的多邊形點集
            polygon = [left, upper, right, lower]
            BBox.append(polygon)
    img = Image.open(rootpath+img_path)

    for j in range(len(BBox)):
        # 進行裁剪
        cropped_image = img.crop(BBox[j])

        # 將 filename 中的 ".png" 替換為 "_j.png"
        new_filename = filename.replace(".png", f"_{j}.png")
        # 新的檔案路徑
        new_filepath = os.path.join("plus", directory, new_filename)

        if not os.path.exists(os.path.join(save_rootpath, "plus", directory)):
            os.makedirs(os.path.join(save_rootpath, "plus", directory))

        cropped_image.save(f"{save_rootpath + new_filepath}")

        text_to_insert = f"plus/{directory}/{new_filename} {label}\n"
        # with open(txt_filepath, 'a') as file:
        #     file.write(text_to_insert)
        queue.put(text_to_insert)
    img.close()  # 关闭原始图像文件

if __name__ == '__main__':
    
    start_time = time.time()  # 記錄開始時間

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_rootpath', help='Img save rootpath')
    parser.add_argument('--txt_filepath', help='Img path save filepath')
    parser.add_argument('--start', help='Img rebuild from')
    parser.add_argument('--end', help='Img rebuild end')
    parser.add_argument('--db_eps', help='dbscan eps')
    args = parser.parse_args()
    print(f"Patent {int(args.start)}:{int(args.end)} rebuild Start")

    # 使用 multiprocessing.Manager 创建一个经理对象
    manager = multiprocessing.Manager()
    # 创建共享的队列
    queue = manager.Queue()

    # 启动写入文件的进程
    writer_process = multiprocessing.Process(target=write_to_file, args=(queue, args.txt_filepath))
    writer_process.start()

    with open('./patlist/train_patent_trn.txt') as f:
        lines = f.read().splitlines()
    # 將每一行轉換為 tuple 並存放在列表中
    data_list = [list(line.split()) for line in lines]
    partial_func = partial(New_data_maker, save_rootpath=args.save_rootpath, txt_filepath=args.txt_filepath, db_eps = int(args.db_eps), queue = queue)  # 这里替换为你的其他参数值

    # 使用 multiprocessing.Pool 執行多進程處理
    with multiprocessing.Pool(processes=16) as pool:
        pool.map(partial_func, data_list[int(args.start):int(args.end)])
    
    queue.put(None)
    # writer_process.join()
    pool.close()
    # pool.join()  # 等待所有进程结束

    end_time = time.time()  # 記錄結束時間
    elapsed_time = end_time - start_time
    print("Elapsed Time: {:.2f} seconds".format(elapsed_time))
    print("Rebuild End")
