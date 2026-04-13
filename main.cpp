#include <led-matrix.h>
#include <graphics.h>

#include <opencv2/opencv.hpp>

#include <portaudio.h>
#include <fftw3.h>
#include <mutex>
#if defined(__linux__)
#include <alsa/asoundlib.h>
#endif

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <string>
#include <algorithm>
#include <cctype>
#include <cstdio>
#if defined(__linux__) || defined(__APPLE__)
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#endif


using rgb_matrix::RGBMatrix;
using rgb_matrix::Canvas;
using rgb_matrix::FrameCanvas;

struct AudioFeatures {
    float rms = 0.0f;
    float left = 0.0f;
    float right = 0.0f;
    float low = 0.0f;
    float mid = 0.0f;
    float high = 0.0f;
    float transient = 0.0f;
};

#if defined(__linux__) && defined(SUPPRESS_ALSA_WARNINGS)
static void silent_alsa_error_handler(const char*, int, const char*, int, const char*, ...) {}
#endif

class KeyboardInput {
public:
    KeyboardInput() {
#if defined(__linux__) || defined(__APPLE__)
        if (!isatty(STDIN_FILENO)) return;
        enabled_ = true;
        tcgetattr(STDIN_FILENO, &original_);
        termios raw = original_;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
        original_flags_ = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, original_flags_ | O_NONBLOCK);
#endif
    }

    ~KeyboardInput() {
#if defined(__linux__) || defined(__APPLE__)
        if (!enabled_) return;
        tcsetattr(STDIN_FILENO, TCSANOW, &original_);
        fcntl(STDIN_FILENO, F_SETFL, original_flags_);
#endif
    }

    int read_key() {
#if defined(__linux__) || defined(__APPLE__)
        if (!enabled_) return -1;
        unsigned char ch = 0;
        ssize_t n = ::read(STDIN_FILENO, &ch, 1);
        return n == 1 ? (int)ch : -1;
#else
        return -1;
#endif
    }

private:
#if defined(__linux__) || defined(__APPLE__)
    termios original_{};
    int original_flags_ = 0;
#endif
    bool enabled_ = false;
};

class AudioAnalyzer {
public:
    static constexpr int SAMPLE_RATE = 48000;
    static constexpr int FRAMES_PER_BUFFER = 1024;
    static constexpr int PREFERRED_CHANNELS = 2;

    PaStream* stream = nullptr;
    std::mutex mtx;
    int input_channels = 0;
    bool use_alsa_fallback = false;
#if defined(__linux__)
    snd_pcm_t* alsa_capture = nullptr;
    std::thread alsa_thread;
    bool alsa_running = false;
    FILE* arecord_pipe = nullptr;
    std::thread arecord_thread;
    bool arecord_running = false;
#endif

    AudioFeatures current_features;

    float smooth_rms = 0.0f;
    float smooth_left = 0.0f;
    float smooth_right = 0.0f;
    float smooth_low = 0.0f;
    float smooth_mid = 0.0f;
    float smooth_high = 0.0f;

    float prev_low = 0.0f;
    float prev_rms = 0.0f;
    float transient_env = 0.0f;

    static int audioCallback(
        const void* inputBuffer,
        void* outputBuffer,
        unsigned long framesPerBuffer,
        const PaStreamCallbackTimeInfo* timeInfo,
        PaStreamCallbackFlags statusFlags,
        void* userData
    ) {
        AudioAnalyzer* self = static_cast<AudioAnalyzer*>(userData);
        self->processInput((const float*)inputBuffer, framesPerBuffer);
        return paContinue;
    }

    float smooth_value(float current, float target, float attack, float release) {
        if (target > current)
            return current + (target - current) * attack;
        else
            return current + (target - current) * release;
    }

    void processInput(const float* input, unsigned long framesPerBuffer) {
        if (!input || input_channels <= 0) return;

        std::vector<float> mono(framesPerBuffer);
        std::vector<float> left_channel(framesPerBuffer);
        std::vector<float> right_channel(framesPerBuffer);

        for (unsigned long i = 0; i < framesPerBuffer; i++) {
            float l = input[i * input_channels];
            float r = (input_channels > 1) ? input[i * input_channels + 1] : l;
            left_channel[i] = l;
            right_channel[i] = r;
            mono[i] = 0.5f * (l + r);
        }

        // RMS
        float rms = 0.0f;
        float left_rms = 0.0f;
        float right_rms = 0.0f;
        for (auto s : mono) rms += s * s;
        for (auto s : left_channel) left_rms += s * s;
        for (auto s : right_channel) right_rms += s * s;
        rms = std::sqrt(rms / mono.size());
        left_rms = std::sqrt(left_rms / left_channel.size());
        right_rms = std::sqrt(right_rms / right_channel.size());

        // FFT
        int N = (int)mono.size();
        fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (N/2 + 1));
        float* in = (float*)fftwf_malloc(sizeof(float) * N);

        for (int i = 0; i < N; i++) in[i] = mono[i];

        fftwf_plan plan = fftwf_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
        fftwf_execute(plan);

        float low = 0.0f, mid = 0.0f, high = 0.0f;
        int low_count = 0, mid_count = 0, high_count = 0;

        for (int i = 0; i < N/2 + 1; i++) {
            float freq = (float)i * SAMPLE_RATE / N;
            float mag = std::sqrt(out[i][0]*out[i][0] + out[i][1]*out[i][1]);

            if (freq >= 20 && freq < 160) {
                low += mag;
                low_count++;
            }
            else if (freq >= 160 && freq < 2000) {
                mid += mag;
                mid_count++;
            }
            else if (freq >= 2000 && freq < 9000) {
                high += mag;
                high_count++;
            }
        }

        if (low_count > 0) low /= low_count;
        if (mid_count > 0) mid /= mid_count;
        if (high_count > 0) high /= high_count;

        fftwf_destroy_plan(plan);
        fftwf_free(in);
        fftwf_free(out);

        // grezza normalizzazione
        float raw_low = std::min(low / 50.0f, 1.0f);
        float raw_mid = std::min(mid / 30.0f, 1.0f);
        float raw_high = std::min(high / 20.0f, 1.0f);
        float raw_rms = std::min(rms * 8.0f, 1.0f);
        float raw_left = std::min(left_rms * 8.0f, 1.0f);
        float raw_right = std::min(right_rms * 8.0f, 1.0f);

        // smoothing
        smooth_rms  = smooth_value(smooth_rms,  raw_rms,  0.25f, 0.05f);
        smooth_left = smooth_value(smooth_left, raw_left, 0.30f, 0.08f);
        smooth_right = smooth_value(smooth_right, raw_right, 0.30f, 0.08f);
        smooth_low  = smooth_value(smooth_low,  raw_low,  0.35f, 0.08f);
        smooth_mid  = smooth_value(smooth_mid,  raw_mid,  0.28f, 0.07f);
        smooth_high = smooth_value(smooth_high, raw_high, 0.22f, 0.06f);

        // transient
        float low_delta = std::max(0.0f, raw_low - prev_low);
        float rms_delta = std::max(0.0f, raw_rms - prev_rms);

        float transient_raw = std::min((low_delta * 0.7f + rms_delta * 0.3f) * 2.2f, 1.0f);
        transient_env = std::max(transient_raw, transient_env * 0.84f);

        prev_low = raw_low;
        prev_rms = raw_rms;

        std::lock_guard<std::mutex> lock(mtx);
        current_features.rms = smooth_rms;
        current_features.left = smooth_left;
        current_features.right = smooth_right;
        current_features.low = smooth_low;
        current_features.mid = smooth_mid;
        current_features.high = smooth_high;
        current_features.transient = transient_env;
    }

    static std::string to_lower(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return (char)std::tolower(c);
        });
        return value;
    }

#if defined(__linux__)
    bool try_open_arecord_device(const char* device_name) {
        std::string cmd = "arecord -D ";
        cmd += "'";
        cmd += device_name;
        cmd += "' -q -t raw -f S16_LE -c 2 -r 48000";

        arecord_pipe = popen(cmd.c_str(), "r");
        if (!arecord_pipe) {
            std::cerr << "[audio-debug] arecord fallback open failed for " << device_name << std::endl;
            return false;
        }

        input_channels = PREFERRED_CHANNELS;
        arecord_running = true;
        arecord_thread = std::thread([this]() {
            std::vector<int16_t> buffer_i16(FRAMES_PER_BUFFER * input_channels);
            std::vector<float> buffer_f32(FRAMES_PER_BUFFER * input_channels);
            while (arecord_running && arecord_pipe) {
                size_t want = buffer_i16.size();
                size_t got = fread(buffer_i16.data(), sizeof(int16_t), want, arecord_pipe);
                if (got == want) {
                    for (size_t i = 0; i < got; ++i) {
                        buffer_f32[i] = (float)buffer_i16[i] / 32768.0f;
                    }
                    processInput(buffer_f32.data(), FRAMES_PER_BUFFER);
                    continue;
                }
                if (feof(arecord_pipe)) {
                    std::cerr << "[audio-debug] arecord fallback ended for " << device_name << std::endl;
                    break;
                }
                if (ferror(arecord_pipe)) {
                    clearerr(arecord_pipe);
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        });

        std::cout << "Audio input: " << device_name << " [host=arecord-fallback] (" << input_channels << "ch)" << std::endl;
        return true;
    }

    bool open_arecord_fallback() {
        const char* env = std::getenv("ALSA_HW_DEVICE");
        if (env && *env) {
            return try_open_arecord_device(env);
        }

        const std::vector<const char*> candidates = {
            "plughw:TX6,0",
            "plughw:2,0",
            "hw:TX6,0",
            "hw:2,0",
            "plughw:CARD=TX6,DEV=0",
            "hw:CARD=TX6,DEV=0",
        };

        for (const char* candidate : candidates) {
            if (try_open_arecord_device(candidate)) {
                return true;
            }
        }
        return false;
    }

    bool try_open_alsa_device(const char* device_name) {
        int err = snd_pcm_open(&alsa_capture, device_name, SND_PCM_STREAM_CAPTURE, 0);
        if (err < 0) {
            std::cerr << "[audio-debug] ALSA fallback open failed for " << device_name
                      << ": " << snd_strerror(err) << std::endl;
            alsa_capture = nullptr;
            return false;
        }

        err = snd_pcm_set_params(
            alsa_capture,
            SND_PCM_FORMAT_FLOAT_LE,
            SND_PCM_ACCESS_RW_INTERLEAVED,
            PREFERRED_CHANNELS,
            SAMPLE_RATE,
            1,
            20000
        );
        if (err < 0) {
            std::cerr << "[audio-debug] ALSA fallback params failed: " << snd_strerror(err) << std::endl;
            snd_pcm_close(alsa_capture);
            alsa_capture = nullptr;
            return false;
        }

        input_channels = PREFERRED_CHANNELS;
        use_alsa_fallback = true;
        alsa_running = true;
        alsa_thread = std::thread([this]() {
            std::vector<float> buffer(FRAMES_PER_BUFFER * input_channels);
            while (alsa_running) {
                int frames = snd_pcm_readi(alsa_capture, buffer.data(), FRAMES_PER_BUFFER);
                if (frames == -EPIPE) {
                    snd_pcm_prepare(alsa_capture);
                    continue;
                }
                if (frames < 0) {
                    frames = snd_pcm_recover(alsa_capture, frames, 1);
                    if (frames < 0) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        continue;
                    }
                }
                if (frames > 0) {
                    processInput(buffer.data(), (unsigned long)frames);
                }
            }
        });

        std::cout << "Audio input: " << device_name << " [host=ALSA-fallback] (" << input_channels << "ch)" << std::endl;
        return true;
    }

    bool open_alsa_fallback() {
        const char* env = std::getenv("ALSA_HW_DEVICE");
        if (env && *env) {
            return try_open_alsa_device(env);
        }

        const std::vector<const char*> candidates = {
            "hw:CARD=TX6,DEV=0",
            "plughw:CARD=TX6,DEV=0",
            "hw:TX6,0",
            "plughw:TX6,0",
            "hw:2,0",
            "plughw:2,0",
        };

        for (const char* candidate : candidates) {
            if (try_open_alsa_device(candidate)) {
                return true;
            }
        }

        return false;
    }
#endif

    bool audio_debug_enabled() const {
        const char* value = std::getenv("AUDIO_DEBUG");
        if (!value || !*value) return true;
        std::string normalized = to_lower(value);
        return !(normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off");
    }

    void dump_audio_devices() const {
        int device_count = Pa_GetDeviceCount();
        std::cerr << "[audio-debug] PortAudio device count: " << device_count << std::endl;
        if (device_count < 0) {
            std::cerr << "[audio-debug] Pa_GetDeviceCount error: " << Pa_GetErrorText(device_count) << std::endl;
            return;
        }

        int default_input = Pa_GetDefaultInputDevice();
        std::cerr << "[audio-debug] Default input device index: " << default_input << std::endl;

        const char* xdg_runtime = std::getenv("XDG_RUNTIME_DIR");
        const char* pulse_server = std::getenv("PULSE_SERVER");
        const char* dbus_addr = std::getenv("DBUS_SESSION_BUS_ADDRESS");
        std::cerr << "[audio-debug] XDG_RUNTIME_DIR=" << (xdg_runtime ? xdg_runtime : "<unset>") << std::endl;
        std::cerr << "[audio-debug] PULSE_SERVER=" << (pulse_server ? pulse_server : "<unset>") << std::endl;
        std::cerr << "[audio-debug] DBUS_SESSION_BUS_ADDRESS=" << (dbus_addr ? dbus_addr : "<unset>") << std::endl;

        for (int i = 0; i < device_count; ++i) {
            const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
            if (!info) continue;

            const PaHostApiInfo* host = Pa_GetHostApiInfo(info->hostApi);
            std::cerr
                << "[audio-debug] device " << i
                << " name=\"" << info->name << "\""
                << " host=\"" << (host ? host->name : "unknown") << "\""
                << " in=" << info->maxInputChannels
                << " out=" << info->maxOutputChannels
                << " default-rate=" << info->defaultSampleRate;
            if (i == default_input) std::cerr << " [default-input]";
            std::cerr << std::endl;
        }
    }

    int chooseInputDevice(int requestedDeviceIndex) {
        if (requestedDeviceIndex >= 0) return requestedDeviceIndex;

        const char* requested_name = std::getenv("AUDIO_DEVICE_SUBSTRING");
        std::string requested_substring = requested_name ? to_lower(requested_name) : "";

        int device_count = Pa_GetDeviceCount();
        if (device_count < 0) return paNoDevice;

        if (!requested_substring.empty()) {
            for (int i = 0; i < device_count; ++i) {
                const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
                if (!info || info->maxInputChannels <= 0) continue;
                if (to_lower(info->name).find(requested_substring) != std::string::npos) {
                    return i;
                }
            }
        }

        auto contains_any = [&](const std::string& haystack, std::initializer_list<const char*> needles) {
            for (const char* needle : needles) {
                if (haystack.find(to_lower(needle)) != std::string::npos) return true;
            }
            return false;
        };

        int default_device = Pa_GetDefaultInputDevice();
        if (default_device != paNoDevice) {
            const PaDeviceInfo* info = Pa_GetDeviceInfo(default_device);
            if (info && info->maxInputChannels > 0) {
                return default_device;
            }
        }

        for (int i = 0; i < device_count; ++i) {
            const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
            if (!info || info->maxInputChannels <= 0) continue;

            std::string name = to_lower(info->name);
            if (contains_any(name, {"usb", "hw:", "alsa", "audio"})) {
                return i;
            }
        }

        int best_device = paNoDevice;
        int best_channels = -1;

        for (int i = 0; i < device_count; ++i) {
            const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
            if (!info || info->maxInputChannels <= 0) continue;

            if (info->maxInputChannels > best_channels) {
                best_device = i;
                best_channels = info->maxInputChannels;
            }
        }

        return best_device;
    }

    bool start(int inputDeviceIndex = -1) {
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            std::cerr << "PortAudio init error\n";
            return false;
        }

        if (audio_debug_enabled()) {
            dump_audio_devices();
        }

        inputDeviceIndex = chooseInputDevice(inputDeviceIndex);

        if (inputDeviceIndex == paNoDevice) {
#if defined(__linux__)
            if (open_alsa_fallback()) {
                return true;
            }
            if (open_arecord_fallback()) {
                return true;
            }
#endif
            std::cerr << "Nessun input audio trovato\n";
            return false;
        }

        const PaDeviceInfo* info = Pa_GetDeviceInfo(inputDeviceIndex);
        if (!info || info->maxInputChannels <= 0) {
            std::cerr << "Device audio non valido\n";
            return false;
        }

        const PaHostApiInfo* host = Pa_GetHostApiInfo(info->hostApi);

        input_channels = std::min(PREFERRED_CHANNELS, info->maxInputChannels);
        std::cout << "Audio input: " << info->name
                  << " [host=" << (host ? host->name : "unknown") << "]"
                  << " (" << input_channels << "ch)" << std::endl;

        PaStreamParameters inputParams;
        inputParams.device = inputDeviceIndex;
        inputParams.channelCount = input_channels;
        inputParams.sampleFormat = paFloat32;
        inputParams.suggestedLatency = info->defaultLowInputLatency;
        inputParams.hostApiSpecificStreamInfo = nullptr;

        err = Pa_OpenStream(
            &stream,
            &inputParams,
            nullptr,
            SAMPLE_RATE,
            FRAMES_PER_BUFFER,
            paNoFlag,
            audioCallback,
            this
        );

        if (err != paNoError) {
            std::cerr << "Errore apertura stream audio: " << Pa_GetErrorText(err) << "\n";
            if (audio_debug_enabled()) {
                dump_audio_devices();
            }
            return false;
        }

        err = Pa_StartStream(stream);
        if (err != paNoError) {
            std::cerr << "Errore start stream audio: " << Pa_GetErrorText(err) << "\n";
            return false;
        }

        return true;
    }

    AudioFeatures getFeatures() {
        std::lock_guard<std::mutex> lock(mtx);
        return current_features;
    }

    void stop() {
#if defined(__linux__)
        if (arecord_running) {
            arecord_running = false;
        }
        if (arecord_pipe) {
            pclose(arecord_pipe);
            arecord_pipe = nullptr;
        }
        if (arecord_thread.joinable()) arecord_thread.join();
        if (alsa_running) {
            alsa_running = false;
            if (alsa_thread.joinable()) alsa_thread.join();
        }
        if (alsa_capture) {
            snd_pcm_drop(alsa_capture);
            snd_pcm_close(alsa_capture);
            alsa_capture = nullptr;
        }
#endif
        if (stream) {
            Pa_StopStream(stream);
            Pa_CloseStream(stream);
            stream = nullptr;
        }
        input_channels = 0;
        use_alsa_fallback = false;
        Pa_Terminate();
    }
};

class VisualEngine {
public:
    int width, height;
    float time_t = 0.0f;

    cv::Mat base_img;      // RGB
    cv::Mat prev_output;   // RGB
    cv::Mat prev_luma;     // float
    cv::Mat luma;          // float
    cv::Mat edge_map;      // float
    cv::Mat motion_map;    // float

    VisualEngine(const cv::Mat& initial_frame, int w, int h)
        : width(w), height(h)
    {
        base_img = initial_frame.clone();
        prev_output = initial_frame.clone();
        luma = compute_luma(base_img);
        edge_map = compute_edge_map(luma);
        prev_luma = luma.clone();
        motion_map = cv::Mat::zeros(height, width, CV_32F);
    }

    cv::Mat compute_luma(const cv::Mat& img) {
        cv::Mat fimg;
        img.convertTo(fimg, CV_32F);

        std::vector<cv::Mat> ch;
        cv::split(fimg, ch);

        cv::Mat out = 0.299f * ch[0] + 0.587f * ch[1] + 0.114f * ch[2];
        return out;
    }

    cv::Mat blur3(const cv::Mat& in) {
        cv::Mat out;
        cv::blur(in, out, cv::Size(3, 3));
        return out;
    }

    cv::Mat compute_edge_map(const cv::Mat& luma_in) {
        cv::Mat gx, gy;
        cv::Sobel(luma_in, gx, CV_32F, 1, 0, 3);
        cv::Sobel(luma_in, gy, CV_32F, 0, 1, 3);

        cv::Mat mag;
        cv::magnitude(gx, gy, mag);
        mag = blur3(mag);

        double minv, maxv;
        cv::minMaxLoc(mag, &minv, &maxv);
        mag = mag / (float(maxv) + 1e-6f);

        mag = (mag - 0.08f) * 4.5f;
        cv::threshold(mag, mag, 0.0, 1.0, cv::THRESH_TOZERO);
        cv::threshold(mag, mag, 1.0, 1.0, cv::THRESH_TRUNC);

        return mag;
    }

    cv::Mat compute_motion_map(const cv::Mat& current_luma) {
        cv::Mat diff;
        cv::absdiff(current_luma, prev_luma, diff);
        diff = blur3(diff);

        double minv, maxv;
        cv::minMaxLoc(diff, &minv, &maxv);
        diff = diff / (float(maxv) + 1e-6f);

        prev_luma = current_luma.clone();
        return diff;
    }

    cv::Mat apply_red_grade(const cv::Mat& img) {
        cv::Mat out;
        img.convertTo(out, CV_32F);

        std::vector<cv::Mat> ch;
        cv::split(out, ch);

        cv::Mat lum = compute_luma(img) / 255.0f;

        ch[0] = ch[0] * 1.55f + lum * 70.0f; // R
        ch[1] = ch[1] * 0.18f + lum * 8.0f;  // G
        ch[2] = ch[2] * 0.12f + lum * 2.0f;  // B

        cv::merge(ch, out);

        out = (out - 128.0f) * 1.15f + 128.0f;
        cv::threshold(out, out, 255.0, 255.0, cv::THRESH_TRUNC);
        cv::threshold(out, out, 0.0, 0.0, cv::THRESH_TOZERO);

        out.convertTo(out, CV_8U);
        return out;
    }

    cv::Mat get_background_static_mask() {
        cv::Mat static_mask = motion_map < 0.20f;
        cv::Mat non_edge_mask = edge_map < 0.22f;

        cv::Mat local_mean = blur3(luma);
        cv::Mat texture;
        cv::absdiff(luma, local_mean, texture);
        texture = blur3(texture);

        double minv, maxv;
        cv::minMaxLoc(texture, &minv, &maxv);
        texture = texture / (float(maxv) + 1e-6f);

        cv::Mat texture_mask = texture > 0.04f;

        cv::Mat final_mask;
        cv::bitwise_and(static_mask, non_edge_mask, final_mask);
        cv::bitwise_and(final_mask, texture_mask, final_mask);

        return final_mask;
    }

    cv::Mat get_edge_static_mask(float edge_thresh = 0.18f, float motion_thresh = 0.18f) {
        cv::Mat edge_mask = edge_map > edge_thresh;
        cv::Mat static_mask = motion_map < motion_thresh;

        cv::Mat final_mask;
        cv::bitwise_and(edge_mask, static_mask, final_mask);
        return final_mask;
    }

    cv::Mat shift_image(const cv::Mat& img, int dx, int dy) {
        cv::Mat out(img.size(), img.type());
        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                int sx = (x - dx + img.cols) % img.cols;
                int sy = (y - dy + img.rows) % img.rows;
                out.at<cv::Vec3b>(y, x) = img.at<cv::Vec3b>(sy, sx);
            }
        }
        return out;
    }

    cv::Mat background_mass_displacement(const cv::Mat& img, float amount) {
        if (amount < 0.02f) return img.clone();

        cv::Mat out = img.clone();
        cv::Mat bg_mask = get_background_static_mask();

        int dx1 = (int)(std::sin(time_t * 0.9f) * (1 + amount * 6));
        int dy1 = (int)(std::cos(time_t * 0.7f) * (1 + amount * 5));

        int dx2 = (int)(std::sin(time_t * 1.3f + 1.2f) * (1 + amount * 4));
        int dy2 = (int)(std::cos(time_t * 1.1f + 0.7f) * (1 + amount * 4));

        cv::Mat shifted1 = shift_image(img, dx1, dy1);
        cv::Mat shifted2 = shift_image(img, dx2, dy2);

        cv::Mat mixed;
        cv::addWeighted(shifted1, 0.5, shifted2, 0.5, 0.0, mixed);

        mixed.copyTo(out, bg_mask);
        return out;
    }

    cv::Mat contour_displacement_static_only(const cv::Mat& img, float amount) {
        if (amount < 0.02f) return img.clone();

        int dx = (int)(std::sin(time_t * 2.2f) * (2 + amount * 6));
        int dy = (int)(std::cos(time_t * 1.7f) * (2 + amount * 6));

        cv::Mat shifted = shift_image(img, dx, dy);
        cv::Mat final_mask = get_edge_static_mask(0.18f, 0.18f);

        cv::Mat out = img.clone();
        shifted.copyTo(out, final_mask);
        return out;
    }

    cv::Mat edge_boundary_wobble_static_only(const cv::Mat& img, float amount) {
        if (amount < 0.02f) return img.clone();

        cv::Mat edge_mask = get_edge_static_mask(0.17f, 0.18f);
        cv::Mat edge_ring;
        cv::dilate(edge_mask, edge_ring, cv::Mat(), cv::Point(-1, -1), 1);
        cv::Mat inner_edge;
        cv::erode(edge_mask, inner_edge, cv::Mat(), cv::Point(-1, -1), 1);
        cv::bitwise_xor(edge_ring, inner_edge, edge_ring);

        int dx1 = (int)std::lround(std::sin(time_t * 2.4f + 0.7f) * (1 + amount * 4));
        int dy1 = (int)std::lround(std::cos(time_t * 1.9f + 1.3f) * (1 + amount * 3));
        int dx2 = (int)std::lround(std::sin(time_t * 3.7f + 2.1f) * (1 + amount * 5));
        int dy2 = (int)std::lround(std::cos(time_t * 2.8f + 0.2f) * (1 + amount * 2));

        cv::Mat shifted1 = shift_image(img, dx1, dy1);
        cv::Mat shifted2 = shift_image(img, dx2, dy2);
        cv::Mat warped;
        cv::addWeighted(shifted1, 0.55, shifted2, 0.45, 0.0, warped);

        // Local horizontal pixel-sorting on the edge boundary ring only.
        for (int y = 0; y < warped.rows; ++y) {
            int x = 0;
            while (x < warped.cols) {
                if (!edge_ring.at<uchar>(y, x)) {
                    ++x;
                    continue;
                }

                int start = x;
                while (x < warped.cols && edge_ring.at<uchar>(y, x)) ++x;
                int end = x;
                if (end - start < 2) continue;

                std::vector<std::pair<int, cv::Vec3b>> segment;
                segment.reserve(end - start);
                for (int sx = start; sx < end; ++sx) {
                    cv::Vec3b px = warped.at<cv::Vec3b>(y, sx);
                    int score = px[0] * 77 + px[1] * 150 + px[2] * 29;
                    segment.push_back({score, px});
                }

                const bool descending = ((y + start) % 2 == 0);
                std::stable_sort(segment.begin(), segment.end(), [descending](const auto& a, const auto& b) {
                    return descending ? a.first > b.first : a.first < b.first;
                });

                for (int sx = start; sx < end; ++sx) {
                    warped.at<cv::Vec3b>(y, sx) = segment[sx - start].second;
                }
            }
        }

        cv::Mat out = img.clone();
        warped.copyTo(out, edge_ring);

        cv::Mat stable_core = get_edge_static_mask(0.24f, 0.16f);
        img.copyTo(out, stable_core);
        return out;
    }

    cv::Mat edge_rgb_glitch_static_only(const cv::Mat& img, float amount) {
        if (amount < 0.02f) return img.clone();

        cv::Mat out = img.clone();
        cv::Mat final_mask = get_edge_static_mask(0.15f, 0.18f);

        int shift_r_x = (int)(1 + amount * 4);
        int shift_b_x = -(int)(1 + amount * 4);
        int shift_g_y = (int)(std::sin(time_t * 2.8f) * (1 + amount * 3));

        cv::Mat r_shift = shift_image(img, shift_r_x, 0);
        cv::Mat g_shift = shift_image(img, 0, shift_g_y);
        cv::Mat b_shift = shift_image(img, shift_b_x, 0);

        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                if (final_mask.at<uchar>(y, x)) {
                    out.at<cv::Vec3b>(y, x)[0] = r_shift.at<cv::Vec3b>(y, x)[0];
                    out.at<cv::Vec3b>(y, x)[1] = g_shift.at<cv::Vec3b>(y, x)[1];
                    out.at<cv::Vec3b>(y, x)[2] = b_shift.at<cv::Vec3b>(y, x)[2];
                }
            }
        }

        return out;
    }

    cv::Mat edge_color_burn_static_only(const cv::Mat& img, float amount) {
        if (amount < 0.02f) return img.clone();

        cv::Mat out;
        img.convertTo(out, CV_32F);

        cv::Mat final_mask = get_edge_static_mask(0.14f, 0.18f);

        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                if (final_mask.at<uchar>(y, x)) {
                    float e = edge_map.at<float>(y, x) * amount;

                    out.at<cv::Vec3f>(y, x)[0] += e * 180.0f;
                    out.at<cv::Vec3f>(y, x)[1] += e * 80.0f;
                    out.at<cv::Vec3f>(y, x)[2] += e * 200.0f;
                }
            }
        }

        cv::threshold(out, out, 255.0, 255.0, cv::THRESH_TRUNC);
        cv::threshold(out, out, 0.0, 0.0, cv::THRESH_TOZERO);

        out.convertTo(out, CV_8U);
        return out;
    }

    cv::Mat preserve_moving_areas(const cv::Mat& img) {
        cv::Mat out;
        img.convertTo(out, CV_32F);

        cv::Mat basef;
        base_img.convertTo(basef, CV_32F);

        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                float m = std::min(1.0f, motion_map.at<float>(y, x) * 2.5f);
                float alpha = m * 0.45f;

                cv::Vec3f a = out.at<cv::Vec3f>(y, x);
                cv::Vec3f b = basef.at<cv::Vec3f>(y, x);

                out.at<cv::Vec3f>(y, x) = a * (1.0f - alpha) + b * alpha;
            }
        }

        out.convertTo(out, CV_8U);
        return out;
    }

    cv::Mat preserve_stillness(const cv::Mat& img, float rms) {
        if (rms > 0.05f) return img.clone();

        float alpha = std::clamp((0.05f - rms) / 0.05f, 0.0f, 1.0f) * 0.80f;

        cv::Mat out;
        cv::addWeighted(img, 1.0f - alpha, base_img, alpha, 0.0, out);
        return out;
    }

    cv::Mat datamosh_kick(const cv::Mat& img, float amount) {
        if (amount < 0.04f || prev_output.empty()) return img.clone();

        cv::Mat out = img.clone();
        cv::Mat displaced_prev = prev_output.clone();

        const int band_h = std::max(2, (int)(2 + amount * 10.0f));
        const int max_shift = std::max(2, (int)(2 + amount * 18.0f));

        for (int y = 0; y < img.rows; y += band_h) {
            int h = std::min(band_h, img.rows - y);
            int shift = (int)std::lround(std::sin(time_t * 3.5f + y * 0.11f) * max_shift);
            cv::Rect band_rect(0, y, img.cols, h);
            cv::Mat band = prev_output(band_rect);
            cv::Mat shifted = shift_image(band, shift, 0);
            shifted.copyTo(displaced_prev(band_rect));
        }

        cv::Mat edge_mask = edge_map > std::max(0.10f, 0.22f - amount * 0.08f);
        cv::Mat motion_mask = motion_map > std::max(0.06f, 0.18f - amount * 0.10f);
        cv::Mat mash_mask;
        cv::bitwise_or(edge_mask, motion_mask, mash_mask);

        cv::Mat blended;
        cv::addWeighted(img, 1.0f - std::min(0.78f, amount * 0.70f),
                        displaced_prev, std::min(0.78f, amount * 0.70f),
                        0.0, blended);
        blended.copyTo(out, mash_mask);

        return out;
    }

    cv::Mat pixel_sort(const cv::Mat& img, float amount, float left_level, float right_level,
                       float threshold_bias = 0.0f) {
        if (amount < 0.03f) return img.clone();

        cv::Mat out = img.clone();
        cv::Mat luma8;
        cv::cvtColor(img, luma8, cv::COLOR_RGB2GRAY);

        const int max_span = std::max(24, (int)(amount * 92.0f));
        const int min_run = std::max(2, (int)(2 + amount * 8.0f));
        const int left_threshold = std::clamp((int)(1 + threshold_bias * 6.0f + (1.0f - amount) * 7.0f + left_level * 14.0f), 1, 56);
        const int right_threshold = std::clamp((int)(1 + threshold_bias * 6.0f + (1.0f - amount) * 7.0f + right_level * 14.0f), 1, 56);

        auto sort_segment = [&](int fixed, int start, int end, bool vertical, bool descending) {
            if (end - start < min_run) return;

            std::vector<std::pair<int, cv::Vec3b>> segment;
            segment.reserve(end - start);

            for (int pos = start; pos < end; ++pos) {
                cv::Vec3b px = vertical ? out.at<cv::Vec3b>(pos, fixed) : out.at<cv::Vec3b>(fixed, pos);
                int score = px[0] * 77 + px[1] * 150 + px[2] * 29;
                segment.push_back({score, px});
            }

            std::stable_sort(segment.begin(), segment.end(), [descending](const auto& a, const auto& b) {
                return descending ? a.first > b.first : a.first < b.first;
            });

            for (int pos = start; pos < end; ++pos) {
                const cv::Vec3b& px = segment[pos - start].second;
                if (vertical) out.at<cv::Vec3b>(pos, fixed) = px;
                else out.at<cv::Vec3b>(fixed, pos) = px;
            }
        };

        for (int y = 0; y < img.rows; ++y) {
            int x = 0;
            while (x < img.cols) {
                const float pan = (float)x / std::max(1, img.cols - 1);
                const int cutoff = (int)std::lround(left_threshold * (1.0f - pan) + right_threshold * pan);
                const int edge_strength = (int)(edge_map.at<float>(y, x) * 255.0f);
                const int motion_strength = (int)(motion_map.at<float>(y, x) * 255.0f);
                const int strength = luma8.at<uchar>(y, x) + edge_strength + motion_strength / 2;
                const bool edge_active = edge_strength > std::max(8, cutoff / 4);
                if (edge_active && strength > cutoff) {
                    int start = x;
                    while (x < img.cols && x - start < max_span) {
                        const float local_pan = (float)x / std::max(1, img.cols - 1);
                        const int local_cutoff = (int)std::lround(left_threshold * (1.0f - local_pan) + right_threshold * local_pan);
                        const int local_edge = (int)(edge_map.at<float>(y, x) * 255.0f);
                        const int local_motion = (int)(motion_map.at<float>(y, x) * 255.0f);
                        const int current = luma8.at<uchar>(y, x) + local_edge + local_motion / 2;
                        const bool local_edge_active = local_edge > std::max(8, local_cutoff / 4);
                        if (!local_edge_active || current <= local_cutoff) break;
                        ++x;
                    }
                    const bool descending = ((y + start) % 2 == 0);
                    sort_segment(y, start, x, false, descending);
                } else {
                    ++x;
                }
            }
        }

        return out;
    }

    cv::Mat update(const AudioFeatures& f, float kick_hold = 0.0f) {
        time_t += 0.05f;

        cv::Mat img = base_img.clone();

        const float energy = std::clamp(f.rms * 1.8f + f.transient * 0.8f + f.mid * 0.35f, 0.0f, 1.0f);
        const float clarity_gate = std::clamp((energy - 0.08f) / 0.55f, 0.0f, 1.0f);
        const float motion_amt = (f.low * 0.9f + f.mid * 0.5f) * (0.20f + 0.80f * clarity_gate);
        const float contour_amt = (f.mid * 1.4f + f.low * 0.4f) * clarity_gate;
        const float glitch_amt = (f.high * 1.3f + f.mid * 0.5f) * clarity_gate;
        const float burn_amt = (f.high * 1.1f + f.transient * 0.8f) * (0.15f + 0.85f * clarity_gate);
        const float sort_amt = std::clamp(f.mid * 0.9f + f.high * 0.7f + f.transient * 0.8f, 0.0f, 1.0f) * clarity_gate;
        const float datamosh_amt = std::clamp(f.transient * 1.5f + f.low * 0.9f + f.rms * 0.35f - 0.18f, 0.0f, 1.0f);

        img = background_mass_displacement(img, motion_amt);
        img = edge_boundary_wobble_static_only(img, contour_amt);
        img = edge_rgb_glitch_static_only(img, glitch_amt);
        img = datamosh_kick(img, datamosh_amt);
        img = edge_color_burn_static_only(img, burn_amt);
        img = pixel_sort(img, sort_amt, f.left, f.right, f.low - 0.25f);

        img = preserve_moving_areas(img);
        img = preserve_stillness(img, f.rms);

        if (kick_hold > 0.01f) {
            cv::Mat emphasized;
            const float alpha = std::clamp(0.50f + kick_hold * 0.18f, 0.0f, 0.78f);
            cv::addWeighted(img, 1.0f - alpha, base_img, alpha, 0.0, emphasized);
            img = emphasized;
        }

        if (clarity_gate < 0.45f) {
            cv::Mat clearer;
            float alpha = 0.25f + (0.45f - clarity_gate) * 1.1f;
            alpha = std::clamp(alpha, 0.0f, 0.70f);
            cv::addWeighted(img, 1.0f - alpha, base_img, alpha, 0.0, clearer);
            img = clearer;
        }

        prev_output = img.clone();

        return img;
    }
};

// --------------------------------------------
// Utils
// --------------------------------------------
bool is_black_frame(const cv::Mat& frame, int threshold = 28, float dark_ratio = 0.985f) {
    cv::Mat luma;
    cv::cvtColor(frame, luma, cv::COLOR_RGB2GRAY);

    int dark = 0;
    int bright = 0;
    int total = luma.rows * luma.cols;

    for (int y = 0; y < luma.rows; y++) {
        for (int x = 0; x < luma.cols; x++) {
            uchar value = luma.at<uchar>(y, x);
            if (value < threshold) dark++;
            if (value > threshold + 22) bright++;
        }
    }

    const float dark_portion = (float)dark / total;
    const float bright_portion = (float)bright / total;

    // Considera "nero" solo un frame quasi totalmente buio, non uno con un riquadro nero.
    return dark_portion > dark_ratio && bright_portion < 0.008f;
}

int env_to_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;

    try {
        return std::stoi(value);
    } catch (...) {
        return fallback;
    }
}

bool env_to_bool(const char* name, bool fallback = false) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;

    std::string normalized = AudioAnalyzer::to_lower(value);
    if (normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on") return true;
    if (normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off") return false;
    return fallback;
}

cv::Mat load_rgb_image_or_blank(const std::string& path, int width, int height) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        img = cv::Mat(height, width, CV_8UC3, cv::Scalar(18, 18, 18));
    } else {
        cv::resize(img, img, cv::Size(width, height));
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    return img;
}

cv::Mat make_orientation_guide(const cv::Mat& base) {
    cv::Mat guide = base.clone();
    const int panel_w = guide.cols / 2;
    const int panel_h = guide.rows / 2;

    cv::rectangle(guide, cv::Rect(0, 0, panel_w, panel_h), cv::Scalar(255, 60, 60), 2);
    cv::rectangle(guide, cv::Rect(panel_w, 0, panel_w, panel_h), cv::Scalar(60, 255, 60), 2);
    cv::rectangle(guide, cv::Rect(0, panel_h, panel_w, panel_h), cv::Scalar(60, 160, 255), 2);
    cv::rectangle(guide, cv::Rect(panel_w, panel_h, panel_w, panel_h), cv::Scalar(255, 220, 80), 2);

    cv::line(guide, cv::Point(panel_w, 0), cv::Point(panel_w, guide.rows), cv::Scalar(255, 255, 255), 1);
    cv::line(guide, cv::Point(0, panel_h), cv::Point(guide.cols, panel_h), cv::Scalar(255, 255, 255), 1);

    auto label_panel = [&](const std::string& text, int x0, int y0, cv::Scalar color, int arrow_dx, int arrow_dy) {
        cv::putText(guide, text, cv::Point(x0 + 6, y0 + 18), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(guide, text, cv::Point(x0 + 6, y0 + 18), cv::FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv::LINE_AA);
        cv::Point center(x0 + panel_w / 2, y0 + panel_h / 2);
        cv::arrowedLine(guide, center, cv::Point(center.x + arrow_dx, center.y + arrow_dy), color, 2, cv::LINE_AA, 0, 0.25);
        cv::circle(guide, center, 5, color, -1, cv::LINE_AA);
    };

    label_panel("P1 TL R270", 0, 0, cv::Scalar(255, 60, 60), 0, -18);
    label_panel("P2 TR R90", panel_w, 0, cv::Scalar(60, 255, 60), 0, 18);
    label_panel("P3 BL R270", 0, panel_h, cv::Scalar(60, 160, 255), 0, -18);
    label_panel("P4 BR R90", panel_w, panel_h, cv::Scalar(255, 220, 80), 0, 18);

    cv::putText(guide, "CHAIN: P3 P1 P2 P4", cv::Point(6, guide.rows - 10), cv::FONT_HERSHEY_SIMPLEX,
                0.45, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(guide, "CHAIN: P3 P1 P2 P4", cv::Point(6, guide.rows - 10), cv::FONT_HERSHEY_SIMPLEX,
                0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    return guide;
}

std::vector<cv::Mat> preload_random_frames(const std::string& video_path, int width, int height, int count = 80) {
    std::vector<cv::Mat> frames;
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        std::cerr << "Errore apertura video.\n";
        return frames;
    }

    int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    if (total_frames <= 0) return frames;

    for (int i = 0; i < count; i++) {
        int idx = rand() % total_frames;
        cap.set(cv::CAP_PROP_POS_FRAMES, idx);

        cv::Mat frame;
        if (!cap.read(frame)) continue;

        cv::resize(frame, frame, cv::Size(width, height));
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        if (!is_black_frame(frame)) {
            frames.push_back(frame.clone());
        }
    }

    return frames;
}

void draw_to_matrix(Canvas* canvas, const cv::Mat& frame) {
    for (int y = 0; y < frame.rows; y++) {
        for (int x = 0; x < frame.cols; x++) {
            cv::Vec3b px = frame.at<cv::Vec3b>(y, x);
            canvas->SetPixel(x, y, px[0], px[1], px[2]);
        }
    }
}

cv::Mat apply_panel_transform(const cv::Mat& panel, int rotate_deg, bool flip_x, bool flip_y) {
    cv::Mat transformed = panel.clone();

    const int normalized = ((rotate_deg % 360) + 360) % 360;
    if (normalized == 90) {
        cv::rotate(transformed, transformed, cv::ROTATE_90_CLOCKWISE);
    } else if (normalized == 180) {
        cv::rotate(transformed, transformed, cv::ROTATE_180);
    } else if (normalized == 270) {
        cv::rotate(transformed, transformed, cv::ROTATE_90_COUNTERCLOCKWISE);
    }

    if (flip_x && flip_y) {
        cv::flip(transformed, transformed, -1);
    } else if (flip_x) {
        cv::flip(transformed, transformed, 1);
    } else if (flip_y) {
        cv::flip(transformed, transformed, 0);
    }

    return transformed;
}

void draw_layout_to_matrix(Canvas* canvas, const cv::Mat& logical_frame, int orientation_variant = 0) {
    const int panel_w = logical_frame.cols / 2;
    const int panel_h = logical_frame.rows / 2;

    const cv::Rect src_p1(0, 0, panel_w, panel_h);
    const cv::Rect src_p2(panel_w, 0, panel_w, panel_h);
    const cv::Rect src_p3(0, panel_h, panel_w, panel_h);
    const cv::Rect src_p4(panel_w, panel_h, panel_w, panel_h);

    struct PanelRoute {
        const char* name;
        cv::Rect source;
        int rotate;
        bool flip_x;
        bool flip_y;
    };

    const PanelRoute normal_routes[] = {
        {"p3", src_p3, 90, false, false},
        {"p1", src_p1, 90, false, false},
        {"p2", src_p2, 270, false, false},
        {"p4", src_p4, 270, false, false},
    };
    const PanelRoute alt_routes[] = {
        {"p3", src_p3, 270, false, false},
        {"p1", src_p1, 270, false, false},
        {"p2", src_p2, 90, false, false},
        {"p4", src_p4, 90, false, false},
    };
    const PanelRoute alt_flip_routes[] = {
        {"p3", src_p3, 270, true, false},
        {"p1", src_p1, 270, true, false},
        {"p2", src_p2, 90, true, false},
        {"p4", src_p4, 90, true, false},
    };

    const PanelRoute* routes = normal_routes;
    if (orientation_variant == 1) routes = alt_routes;
    else if (orientation_variant == 2) routes = alt_flip_routes;

    for (int slot = 0; slot < 4; ++slot) {
        cv::Mat panel = apply_panel_transform(logical_frame(routes[slot].source), routes[slot].rotate,
                                              routes[slot].flip_x, routes[slot].flip_y);
        const int dst_x = slot * panel_w;
        for (int y = 0; y < panel.rows; ++y) {
            for (int x = 0; x < panel.cols; ++x) {
                cv::Vec3b px = panel.at<cv::Vec3b>(y, x);
                canvas->SetPixel(dst_x + x, y, px[0], px[1], px[2]);
            }
        }
    }
}

// --------------------------------------------
// MAIN
// --------------------------------------------
int main(int argc, char *argv[]) {
    srand(time(nullptr));

    // Canvas fisico: 4 pannelli 64x64 in chain lineare.
    // Contenuto logico: immagine 128x128 suddivisa in 4 quadranti.
    const int PANEL_ROWS = env_to_int("MATRIX_ROWS", 64);
    const int PANEL_COLS = env_to_int("MATRIX_COLS", 64);
    const int CHAIN_LENGTH = env_to_int("MATRIX_CHAIN", 4);
    const int PARALLEL = env_to_int("MATRIX_PARALLEL", 1);
    const int LOGICAL_WIDTH = PANEL_COLS * 2;
    const int LOGICAL_HEIGHT = PANEL_ROWS * 2;
    const std::string VIDEO_PATH = "video.mp4";

#if defined(__linux__) && defined(SUPPRESS_ALSA_WARNINGS)
    // PortAudio/ALSA prova diversi PCM durante la discovery; disattiviamo il rumore su stderr.
    snd_lib_error_set_handler(silent_alsa_error_handler);
#endif

    rgb_matrix::RGBMatrix::Options defaults;
    defaults.rows = PANEL_ROWS;
    defaults.cols = PANEL_COLS;
    defaults.chain_length = CHAIN_LENGTH;
    defaults.parallel = PARALLEL;
    defaults.hardware_mapping = "regular";
    defaults.brightness = env_to_int("MATRIX_BRIGHTNESS", 70);
    defaults.disable_hardware_pulsing = true;
    defaults.row_address_type = env_to_int("MATRIX_ROW_ADDR_TYPE", 0);
    defaults.multiplexing = env_to_int("MATRIX_MULTIPLEXING", 0);

    rgb_matrix::RuntimeOptions runtime;
    runtime.gpio_slowdown = env_to_int("MATRIX_GPIO_SLOWDOWN", 4);

    RGBMatrix *matrix = rgb_matrix::CreateMatrixFromOptions(defaults, runtime);
    if (matrix == nullptr) return 1;

    FrameCanvas *offscreen = matrix->CreateFrameCanvas();
    const bool test_mode = env_to_bool("MATRIX_TEST_MODE", false);
    const std::string TEST_IMAGE_PATH = "base.jpg";

    if (test_mode) {
        cv::Mat base = load_rgb_image_or_blank(TEST_IMAGE_PATH, LOGICAL_WIDTH, LOGICAL_HEIGHT);
        cv::Mat guide = make_orientation_guide(base);
        while (true) {
            draw_layout_to_matrix(offscreen, guide, 0);
            offscreen = matrix->SwapOnVSync(offscreen);
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    }

    cv::VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) {
        std::cerr << "Impossibile aprire video: " << VIDEO_PATH << "\n";
        return 1;
    }

    std::vector<cv::Mat> random_buffer = preload_random_frames(VIDEO_PATH, LOGICAL_WIDTH, LOGICAL_HEIGHT, 100);
    if (random_buffer.empty()) {
        std::cerr << "Buffer random vuoto.\n";
    }

    cv::Mat frame;
    cap.read(frame);
    cv::resize(frame, frame, cv::Size(LOGICAL_WIDTH, LOGICAL_HEIGHT));
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

    VisualEngine visual(frame, LOGICAL_WIDTH, LOGICAL_HEIGHT);

    AudioAnalyzer audio;
    if (!audio.start()) {
        std::cerr << "Errore avvio audio analyzer.\n";
        return 1;
    }

    KeyboardInput keyboard;
    bool paused = false;

    auto last_kick = std::chrono::steady_clock::now() - std::chrono::milliseconds(500);
    cv::Mat active_kick_frame;
    auto kick_frame_until = std::chrono::steady_clock::time_point::min();
    int active_orientation_variant = 0;
    auto orientation_variant_until = std::chrono::steady_clock::time_point::min();
    auto next_random_jump = std::chrono::steady_clock::now() + std::chrono::seconds(240 + rand() % 121);

    while (true) {
        for (int key = keyboard.read_key(); key != -1; key = keyboard.read_key()) {
            if (key == 'p' || key == 'P' || key == ' ') {
                paused = !paused;
                std::cout << (paused ? "Playback paused" : "Playback resumed") << std::endl;
            } else if (key == 'q' || key == 'Q') {
                audio.stop();
                delete matrix;
                return 0;
            }
        }

        AudioFeatures features = audio.getFeatures();
        // --------------------------------
        // KICK JUMP con cooldown
        // --------------------------------
        auto now = std::chrono::steady_clock::now();
        float since_kick = std::chrono::duration<float>(now - last_kick).count();

        const bool kick_detected =
            (features.transient > 0.08f && features.low > 0.18f) ||
            (features.transient > 0.06f && features.mid > 0.16f) ||
            (features.rms > 0.08f && features.mid > 0.14f) ||
            (features.rms > 0.12f);
        if (kick_detected && since_kick > 0.45f) {
            if (!random_buffer.empty()) {
                int best_idx = rand() % random_buffer.size();
                double best_score = -1.0;
                for (int attempt = 0; attempt < 18; ++attempt) {
                    int idx = rand() % random_buffer.size();
                    cv::Mat diff;
                    cv::absdiff(frame, random_buffer[idx], diff);
                    cv::Scalar score = cv::mean(diff);
                    double total = score[0] + score[1] + score[2];
                if (total > best_score) {
                        best_score = total;
                        best_idx = idx;
                    }
                }
                active_kick_frame = random_buffer[best_idx].clone();
                kick_frame_until = now + std::chrono::milliseconds(45);
            }
            last_kick = now;
        }

        if (!random_buffer.empty() && now >= next_random_jump) {
            active_kick_frame = random_buffer[rand() % random_buffer.size()].clone();
            kick_frame_until = now + std::chrono::milliseconds(90);
            next_random_jump = now + std::chrono::seconds(240 + rand() % 121);
        }

        // --------------------------------
        // Playback normale
        // --------------------------------
        if (!paused) {
            if (!cap.read(frame)) {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                cap.read(frame);
            }
            cv::resize(frame, frame, cv::Size(LOGICAL_WIDTH, LOGICAL_HEIGHT));
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        }

        if (is_black_frame(frame) && !random_buffer.empty()) {
            int idx = rand() % random_buffer.size();
            frame = random_buffer[idx].clone();
        }

        const bool kick_frame_active = !active_kick_frame.empty() && now < kick_frame_until;
        float kick_hold = 0.0f;
        if (kick_frame_active) {
            float remaining = std::chrono::duration<float>(kick_frame_until - now).count();
            kick_hold = std::clamp(remaining / 0.045f, 0.0f, 1.0f);
        }
        if (kick_frame_active) {
            visual.base_img = active_kick_frame.clone();
        } else {
            visual.base_img = frame.clone();
        }
        visual.luma = visual.compute_luma(visual.base_img);
        visual.edge_map = visual.compute_edge_map(visual.luma);
        visual.motion_map = visual.compute_motion_map(visual.luma);

        cv::Mat out = visual.update(features, kick_hold);

        const float orientation_energy = std::clamp(features.transient * 1.75f + features.high * 1.05f + features.rms * 0.60f, 0.0f, 1.0f);
        if (orientation_energy > 0.50f && since_kick > 0.30f) {
            active_orientation_variant = 1 + (rand() % 2);
            orientation_variant_until = now + std::chrono::milliseconds(180);
        }
        if (now >= orientation_variant_until) {
            active_orientation_variant = 0;
        }

        draw_layout_to_matrix(offscreen, out, active_orientation_variant);
        offscreen = matrix->SwapOnVSync(offscreen);

        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60fps target
    }

    audio.stop();
    delete matrix;
    return 0;
}
