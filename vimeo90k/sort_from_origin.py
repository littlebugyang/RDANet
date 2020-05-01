import os
import shutil
from PIL import Image
from numpy import average, dot, linalg

# 对图片进行统一化处理
def get_thum(image, size=(64, 64), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image


# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数？？
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res
    
# 输出可用图片路径
def output_available_list(n_frame, root):
    # n_frame 为符合数据入口处要求输入的 nFrames，默认为 7
    seq_num = len(os.listdir(root)) - 78
    # 以上78为原有数据集的78个文件夹
    output_file = open('./vimeo90k/available_list.txt', mode='w')
    
    for i in range(seq_num):
        seq_dir = os.path.join(root, 'seq_' + str(i+1).zfill(7))
        frame_in_seq = os.listdir(seq_dir)
        frame_in_seq.sort()
        # 不符合训练要求
        if len(frame_in_seq) < n_frame:
            continue
    
        # half = int(n_frame / 2)
        # if n_frame % 2 == 0:
        #     # n_frame 包括目标帧在内为偶数帧，则：前取的帧数与后取的帧数一致
        #     available_frame_path = frame_in_seq[half:-(half-1)]
        # else:
        #     # n_frame 包括目标帧在内为奇数帧
        #     available_frame_path = frame_in_seq[half:-(half-0)]
    
        # 根据上面注释代码缩写
        half = int(n_frame / 2)
        available_frame_path = frame_in_seq[half:1- (n_frame % 2) - half]
        print(seq_dir, len(frame_in_seq), available_frame_path)
        for frame_path in available_frame_path:
            output_file.write(os.path.join(seq_dir, frame_path) + '\n')
    
    output_file.close()

seq_root = "./vimeo90k/sequences/"
scene_num = 78

# 假设原来每个seq文件夹中的帧数都为n_frame_in_seq
n_frame_in_seq = 3

seq_count = 0
img_count = 0

if n_frame_in_seq < 3:
    print("工作很难进行下去！")
    exit(1)

last_img_paths = ['./vimeo90k/default.png'] * n_frame_in_seq

seq_dir = ''

for i in range(scene_num):
    scene_dir = os.path.join(seq_root, str(i + 1).zfill(5))

    scene_sub_dir_list = os.listdir(scene_dir)
    scene_sub_dir_list.sort()

    for j, sub_dir in enumerate(scene_sub_dir_list):
        scene_sub_dir = os.path.join(scene_dir, sub_dir)
        scene_sub_dir_list = os.listdir(scene_sub_dir)
        scene_sub_dir_list.sort()

        skip_times = 0

        # 发现该seq中的首帧与上个seq中的第2帧一样，当前seq文件夹中可以跳几个帧
        if last_img_paths[1] != './vimeo90k/default.png':
            cos_diff = image_similarity_vectors_via_numpy(Image.open(os.path.join(scene_sub_dir, 'im1.png')),
                                                          Image.open(last_img_paths[1]))
            if int(cos_diff) == 1 or cos_diff > 0.99:
                skip_times = n_frame_in_seq - 1

        for name in scene_sub_dir_list:
            source_path = os.path.join(scene_sub_dir, name)
            last_img_paths[0:-1] = last_img_paths[1:]
            last_img_paths[-1] = source_path

            if skip_times != 0:
                skip_times -= 1
                continue

            cos_diff = image_similarity_vectors_via_numpy(Image.open(last_img_paths[-2]), Image.open(source_path))

            if cos_diff < 0.94:
                seq_dir = os.path.join(seq_root, 'seq_' + str(seq_count + 1).zfill(7))
                if os.path.exists(seq_dir):
                    shutil.rmtree(seq_dir)
                os.mkdir(seq_dir)
                seq_count += 1
                img_count = 0

            target_path = os.path.join(seq_dir, 'im' + str(img_count + 1).zfill(5) + '.png')
            img_count += 1
            shutil.copy(source_path, target_path)
            print(source_path, target_path)

print('Concatenation done')

output_available_list(7, seq_root)
