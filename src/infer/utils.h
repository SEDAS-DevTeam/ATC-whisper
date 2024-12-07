#include <iostream>
#include <unistd.h>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <cstdlib>
#include <filesystem>
#include <queue>
#include <array>
#include <algorithm>
#include <csignal>

// definitions/aliases
namespace fs = std::filesystem;
bool running = true;
const std::string main_path = fs::current_path().u8string() + "/";

class SEDThread {
    public:

        bool running = true;
        bool thread_stop = true;

        void start(){ thread_stop = false; }
        void stop(){ running = false; }
        void pause(){ thread_stop = true; }
};


std::string execute_command(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        result += buffer.data();
        std::cout << buffer.data();
    }
    return result;
}

int rand_choice(uint32_t npos){
    srand(time(NULL));
    return rand() % npos;
}

bool search_vector(std::vector<std::string> vec, std::string input){
    return std::find(vec.begin(), vec.end(), input) != vec.end();
}

bool search_string(std::string str, std::string substr){
    return str.find(substr) != std::string::npos;
}