# Import Libraries
import numpy as np
import cv2
import argparse
import time

# Import User Libraries
import L0_helpers

# Image File Path
image_r = "images/flowers.jpg"
image_w = "out_serial.png"

# L0 minimization parameters
kappa = 2.0;
_lambda = 2e-2;

# Verbose output
verbose = False;

def parse_args():
    parser = argparse.ArgumentParser(
            description="Serial implementation of image smoothing via L0 gradient minimization")
    parser.add_argument('-k', type=float, default=2.0,
                        metavar='kappa', help='updating weight (default 2.0)')
    parser.add_argument('-l', type=float, default=2e-2,
                        metavar='lambda', help='smoothing weight (default 2e-2)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='enable verbose logging for each iteration')

    args = parser.parse_args()
    return args

def image_smoothing_via_L0_gradient_minimization(nor_img):
    # L0 minimization parameters
    kappa = args.k
    _lambda = args.l

    verbose = True  # Verbose output

    # Timers
    step_1 = 0.0
    step_2 = 0.0
    step_2_fft = 0.0

    # Start time
    start_time = time.time()

    # Validate image format
    height, width, channel = np.int32(nor_img.shape)
    assert channel == 3, "Error: input must be 3-channel RGB image"
    print("Processing %d x %d RGB image" % (height, width))

    # Initialize S as I
    S = np.float32(noise_img) #/ 256

    # Compute image OTF
    size_2D = [height, width]
    fx = np.int32([[1, -1]])
    fy = np.int32([[1], [-1]])
    otfFx = L0_helpers.psf2otf(fx, size_2D)
    otfFy = L0_helpers.psf2otf(fy, size_2D)

    # Compute F(I)
    FI = np.complex64(np.zeros((height, width, channel)))
    FI[:, :, 0] = np.fft.fft2(S[:, :, 0])
    FI[:, :, 1] = np.fft.fft2(S[:, :, 1])
    FI[:, :, 2] = np.fft.fft2(S[:, :, 2])

    # Compute MTF
    MTF = np.power(np.abs(otfFx), 2) + np.power(np.abs(otfFy), 2)
    MTF = np.tile(MTF[:, :, np.newaxis], (1, 1, channel))

    # Initialize buffers
    h = np.float32(np.zeros((height, width, channel)))
    v = np.float32(np.zeros((height, width, channel)))
    dxhp = np.float32(np.zeros((height, width, channel)))
    dyvp = np.float32(np.zeros((height, width, channel)))
    FS = np.complex64(np.zeros((height, width, channel)))

    # Iteration settings
    beta_max = 1e5
    beta = 2 * _lambda
    iteration = 0

    # Done initializing
    init_time = time.time()

    # Iterate until desired convergence in similarity
    while beta < beta_max:

        if verbose:
            print("ITERATION %i" % iteration)

        ### Step 1: estimate (h, v) subproblem

        # subproblem 1 start time
        s_time = time.time()

        # compute dxSp
        h[:, 0:width - 1, :] = np.diff(S, 1, 1)
        h[:, width - 1:width, :] = S[:, 0:1, :] - S[:, width - 1:width, :]

        # compute dySp
        v[0:height - 1, :, :] = np.diff(S, 1, 0)
        v[height - 1:height, :, :] = S[0:1, :, :] - S[height - 1:height, :, :]

        # compute minimum energy E = dxSp^2 + dySp^2 <= _lambda/beta
        t = np.sum(np.power(h, 2) + np.power(v, 2), axis=2) < _lambda / beta
        t = np.tile(t[:, :, np.newaxis], (1, 1, 3))

        # compute piecewise solution for hp, vp
        h[t] = 0
        v[t] = 0

        # subproblem 1 end time
        e_time = time.time()
        step_1 = step_1 + e_time - s_time
        if verbose:
            print("-subproblem 1: estimate (h,v)")
            print("--time: %f (s)" % (e_time - s_time))

        ### Step 2: estimate S subproblem

        # subproblem 2 start time
        s_time = time.time()

        # compute dxhp + dyvp
        dxhp[:, 0:1, :] = h[:, width - 1:width, :] - h[:, 0:1, :]
        dxhp[:, 1:width, :] = -(np.diff(h, 1, 1))
        dyvp[0:1, :, :] = v[height - 1:height, :, :] - v[0:1, :, :]
        dyvp[1:height, :, :] = -(np.diff(v, 1, 0))
        normin = dxhp + dyvp

        fft_s = time.time()
        FS[:, :, 0] = np.fft.fft2(normin[:, :, 0])
        FS[:, :, 1] = np.fft.fft2(normin[:, :, 1])
        FS[:, :, 2] = np.fft.fft2(normin[:, :, 2])
        fft_e = time.time()
        step_2_fft += fft_e - fft_s

        # solve for S + 1 in Fourier domain
        denorm = 1 + beta * MTF
        print(denorm)
        print(((FI + beta * FS) / denorm).shape)
        FS = (FI + beta * FS) / denorm

        # inverse FFT to compute S + 1
        fft_s = time.time()
        S[:, :, 0] = np.float32((np.fft.ifft2(FS[:, :, 0])).real)
        S[:, :, 1] = np.float32((np.fft.ifft2(FS[:, :, 1])).real)
        S[:, :, 2] = np.float32((np.fft.ifft2(FS[:, :, 2])).real)
        fft_e = time.time()
        step_2_fft += fft_e - fft_s

        # subproblem 2 end time
        e_time = time.time()
        step_2 = step_2 + e_time - s_time
        if verbose:
            print("-subproblem 2: estimate S + 1")
            print("--time: %f (s)" % (e_time - s_time))
            print("")

        # update beta for next iteration
        beta *= kappa
        iteration += 1

    # S = S * 256  # Rescale image

    final_time = time.time()  # Total end time

    print("Total Time: %f (s)" % (final_time - start_time))
    print("Setup: %f (s)" % (init_time - start_time))
    print("Step 1: %f (s)" % (step_1))
    print("Step 2: %f (s)" % (step_2))
    print("Step 2 (FFT): %f (s)" % (step_2_fft))
    print("Iterations: %d" % (iteration))
    
    return S

def plot_image(image, image_title, is_axis=False):
    """
    展示图像
    :param image: 展示的图像，一般是 np.array 类型
    :param image_title: 展示图像的名称
    :param is_axis: 是否需要关闭坐标轴，默认展示坐标轴
    :return:
    """
    # 展示图片
    plt.imshow(image)
    
    # 关闭坐标轴,默认关闭
    if not is_axis:
        plt.axis('off')

    # 展示受损图片的名称
    plt.title(image_title)

    # 展示图片
    plt.show()

def save_image(filename, image):
    """
    将np.ndarray 图像矩阵保存为一张 png 或 jpg 等格式的图片
    :param filename: 图片保存路径及图片名称和格式
    :param image: 图像矩阵，一般为np.array
    :return:
    """
    # np.copy() 函数创建一个副本。
    # 对副本数据进行修改，不会影响到原始数据，它们物理内存不在同一位置。
    img = np.copy(image)
    
    # 从给定数组的形状中删除一维的条目
    img = img.squeeze()
    
    # 将图片数据存储类型改为 np.uint8
    if img.dtype == np.double:
        
        # 若img数据存储类型是 np.double ,则转化为 np.uint8 形式
        img = img * np.iinfo(np.uint8).max
        
        # 转换图片数组数据类型
        img = img.astype(np.uint8)
    
    # 将 RGB 方式转换为 BGR 方式
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 生成图片
    cv2.imwrite(filename, img)

def normalization(image):
    """
    将数据线性归一化
    :param image: 图片矩阵，一般是np.array 类型 
    :return: 将归一化后的数据，在（0,1）之间
    """
    # 获取图片数据类型对象的最大值和最小值
    info = np.iinfo(image.dtype)
    
    # 图像数组数据放缩在 0-1 之间
    return image.astype(np.double) / info.max

def main():
    img_path = "images/flowers.jpg"
    img = read_image(img_path)

    # 图像数据归一化
    nor_img = normalization(img)
    res_img = image_smoothing_via_L0_gradient_minimization(nor_img)

if __name__ == '__main__':
    args = parse_args()
    main()
