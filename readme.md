# 视觉多线程推理框架

## 简介

使用openvino实现在intel 11代nuc上多路推理海康相机

## 环境准备

考虑到项目编写的时间是2024年，使用目前支持比较好的Ubuntu 22.04

为了方便调试，使用Desktop版

为了后续安装与编译方便，建议开启科学上网，如果不行请参照清华源的帮助文档做一下国内镜像换源

### 依赖库安装

首先是用于监控intel核显与cpu占用的工具
```
sudo apt install -y intel-gpu-tools htop
```

其次是opencv与openvino的依赖，还有boost库
```
sudo apt update && sudo apt install -y git cmake g++ wget unzip curl libboost-all-dev
```

可选的，添加ffmpeg以方便录视频和后续操作
```
sudo apt install -y ffmpeg
```

可选的，安装网络库方便查看ip
```
sudo apt install -y net-tools
```
### 编译opencv实现tbb加速

[编译参照这篇文章](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)

[编译选项看这个，如果你对openmp更熟悉可以换](https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html)
```
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.0.zip
unzip opencv.zip
mkdir -p build && cd build
cmake -D WITH_TBB=ON ..
make -j8
```
这里如果不用科学上网会出现下载tbb库和ippicv库不成功的问题，没有什么太好的解决办法，只能自己看看3rdparty、ippicv文件夹下的cmake文件改一下对应的地址为镜像地址，tbb使用apt提前下好

### 编译openvino实现引用tbb加速的opencv

[编译参照这篇文章](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_linux.md)

在这里和它编写不同的是我们要把git的分支切换到你想要的版本，这里选择的是2024.4
```
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git checkout releases/2024/4
```
初始化依赖看网络
```
# 国内网络执行以下两个脚本使用gitee
chmod +x scripts/submodule_update_with_gitee.sh
./scripts/submodule_update_with_gitee.sh
# 科学上网执行这个就好
git submodule update --init --recursive
```
因为我们使用的是c++，就不做他的python接口了，不然很麻烦，编译选项一般不用动，它默认的就是最大优化了

非要使用他的python相关工具走pip再下一个openvino就好，不然两者环境冲突，你的pip容易崩
```
sudo ./install_build_dependencies.sh
mkdir build && cd build
# 这里<path to OpenCVConfig.cmake>替换为你上文的opencv/build的路径
cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=<path to OpenCVConfig.cmake> ..
# 这里使用make -j8也可以，不过他教程给的这个就用这个了
cmake --build . --parallel 8
```
这样依赖库的准备基本上就完成了

## 项目运行
编译
```
git clone https://github.com/pitlst/toe_aimbot.git
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```
运行的话直接执行对应可执行文件即可

部署的时候记得挂一个开机自启动

看门狗程序保活看你的想法，简单的话写个shell死循环启动就行，我随项目会附一个；复杂的起两个线程互相盯着死没死，死了给拉起来，也会给接口

## 二次开发

### 相机接口

相机类的所属文件为camera/camera.hpp

虽然没有限制，但仍然建议一口气将所有的相机均初始化完成后开始取流，同时留好了多线程的锁，支持线程安全

对于多个相机，应当在

一个类对应一个相机，并在初始化时指定index也就是第几台相机，最多支持7个相机，其实这部分可以使用宏重写以达到支持到海康sdk的极限即256个相机，但是没必要且没有可读性，我就手动宏展开了

实际上在hik_init函数执行完成后，全局变量frame_array里就会实时更新最新的相机帧

如果你不加锁并且多线程去读相机，会出现图像上半部分上一帧，下半部分下一帧的情况，此时不会报错

这里是一个简单的使用demo
```
#include <iostream>
#include <fstream>

#include "camera.hpp"
#include "nlohmann/json.hpp"

int main()
{
    toe::hik_camera temp;
    std::ifstream f("../config.json");
    nlohmann::json temp_apra = nlohmann::json::parse(f);
    int device_num = 0;
    temp.hik_init(temp_apra, device_num);
    int k = 0;
    cv::Mat img;
    while (k != 27)
    {
        mutex_array[device_num].lock();
        img = frame_array[device_num];
        mutex_array[device_num].unlock();
        if (img.data)
        {
            cv::imshow("frame",img);
        }
        k = cv::waitKey(1);
    }
    temp.hik_end();
    return 0;
}
```
对于程序中提到的配置文件，其结构如下
```
    "camera": {
        "0": {
            "width": 864,
            "height": 864,
            "offset_x": 0,
            "offset_y": 0,
            "Reverse_X": false,
            "Reverse_Y": false,
            "exposure": 5000,
            "gain": 10,
            "balck_level": 240
        },
        "1": {
            "width": 1440,
            "height": 1080,
            "offset_x": 0,
            "offset_y": 0,
            "Reverse_X": false,
            "Reverse_Y": false,
            "exposure": 5000,
            "gain": 10,
            "balck_level": 240
        }
    }
```
具体含义就是长宽像素，横方向和纵方向的偏移，是否在xy方向翻转，曝光，增益，黑电平
这些参数的实际效果与含义在海康sdk的文档上写的很清楚，我就不展开了，常用的就这些，有需要提issue我再加


正在编写ing