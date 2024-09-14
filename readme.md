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

### 项目结构



正在编写ing