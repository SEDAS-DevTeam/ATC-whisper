#include "utils.h"
#include "voice_recog.h"

VoiceRecognition voice_recog;

static void signal_handler(int signal){
    if (signal == SIGINT){
        voice_recog.stop();
        running = false;
    }
}

int main(){
    // set sigint for graceful stop
    std::signal(SIGINT, signal_handler);

    voice_recog.load_params(8, 10000, 0.6);
    voice_recog.load_model(); // TODO

    std::thread thread_recog(&VoiceRecognition::run, &voice_recog);
    voice_recog.start();

    /*
        A simple loop to keep everything running
    */
    while (running){
        sleep(1);
    }

    voice_recog.stop();
    thread_recog.join();

    std::cout << "Main program terminated." << std::endl;
    return 0;
}