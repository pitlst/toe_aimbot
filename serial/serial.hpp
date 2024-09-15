#ifndef TOE_SERIAL_H_
#define TOE_SERIAL_H_

#include <vector>
#include <mutex>

#include "debug.hpp"
#include "nlohmann/json.hpp"
#include "structs.hpp"

#include <boost/asio.hpp>

namespace toe
{
    class serial final
    {
    public:
        serial() = default;
        ~serial() = default;
        // 打开串口
        void init_port();
        void init_port(const nlohmann::json & input);
        //返回封装好的模式和颜色信息
        void get_msg(int & color, int & mode);
        //发送封装好的坐标信息
        void send_msg(const std::vector<double> & msg);
    private:
        // boost串口相关  
        boost::asio::io_service io_s;
        boost::asio::serial_port serial_fd = boost::asio::serial_port(io_s);
        boost::system::error_code m_ec;
        // 接收发送缓冲区
        uint8_t rbuff[1024];
        uint8_t sbuff[12];
        // 串口路径
        std::string post_path;
        //记录的模式信息
        int mode = DEFALUTE_MODE;
        //记录的颜色信息
        bool color = DEFALUTE_COLOR;
        // 串口输入的线程锁
        std::mutex input_mutex_;
        // 串口输出的线程锁
        std::mutex output_mutex_;
    };
}

#endif