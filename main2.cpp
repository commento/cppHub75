#include <led-matrix.h>
#include <graphics.h>

#include <opencv2/opencv.hpp>

#include <portaudio.h>
#include <fftw3.h>
#include <mutex>
#if defined(__linux__) && defined(SUPPRESS_ALSA_WARNINGS)
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
#include <array>
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

        int default_device = Pa_GetDefaultInputDevice();
        if (default_device != paNoDevice) {
            const PaDeviceInfo* info = Pa_GetDeviceInfo(default_device);
            if (info && info->maxInputChannels > 0) {
                return default_device;
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

        inputDeviceIndex = chooseInputDevice(inputDeviceIndex);

        if (inputDeviceIndex == paNoDevice) {
            std::cerr << "Nessun input audio trovato\n";
            return false;
        }

        const PaDeviceInfo* info = Pa_GetDeviceInfo(inputDeviceIndex);
        if (!info || info->maxInputChannels <= 0) {
            std::cerr << "Device audio non valido\n";
            return false;
        }

        input_channels = std::min(PREFERRED_CHANNELS, info->maxInputChannels);
        std::cout << "Audio input: " << info->name
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
            std::cerr << "Errore apertura stream audio\n";
            return false;
        }

        err = Pa_StartStream(stream);
        if (err != paNoError) {
            std::cerr << "Errore start stream audio\n";
            return false;
        }

        return true;
    }

    AudioFeatures getFeatures() {
        std::lock_guard<std::mutex> lock(mtx);
        return current_features;
    }

    void stop() {
        if (stream) {
            Pa_StopStream(stream);
            Pa_CloseStream(stream);
            stream = nullptr;
        }
        input_channels = 0;
        Pa_Terminate();
    }
};

class ConstellationEngine {
public:
    static constexpr float kTau = 6.28318530718f;

    struct StarNode {
        cv::Point3f anchor;
        cv::Point3f drift;
        float size = 1.0f;
        float phase = 0.0f;
        float twinkle = 0.0f;
        float depth_bias = 0.0f;
        std::array<float, 3> color {1.0f, 1.0f, 1.0f};
    };

    struct ProjectedStar {
        cv::Point2f pos;
        float depth = 0.0f;
        float intensity = 0.0f;
        float size = 1.0f;
        std::array<float, 3> color {1.0f, 1.0f, 1.0f};
    };

    int width, height;
    float time_t = 0.0f;
    std::vector<StarNode> stars;
    cv::Mat prev_output;

    explicit ConstellationEngine(int w, int h)
        : width(w), height(h), prev_output(h, w, CV_8UC3, cv::Scalar(0, 0, 0))
    {
        seed_stars(52);
    }

    void seed_stars(int count) {
        stars.clear();
        stars.reserve(count);

        for (int i = 0; i < count; ++i) {
            const float radius = 0.18f + ((float) rand() / RAND_MAX) * 1.05f;
            const float azimuth = ((float) rand() / RAND_MAX) * kTau;
            const float elevation = (((float) rand() / RAND_MAX) - 0.5f) * 1.45f;

            StarNode node;
            node.anchor = cv::Point3f(
                std::cos(azimuth) * std::cos(elevation) * radius,
                std::sin(elevation) * radius * 0.85f,
                std::sin(azimuth) * std::cos(elevation) * radius
            );
            node.drift = cv::Point3f(
                (((float) rand() / RAND_MAX) - 0.5f) * 0.45f,
                (((float) rand() / RAND_MAX) - 0.5f) * 0.30f,
                (((float) rand() / RAND_MAX) - 0.5f) * 0.55f
            );
            node.size = 0.8f + ((float) rand() / RAND_MAX) * 2.1f;
            node.phase = ((float) rand() / RAND_MAX) * kTau;
            node.twinkle = 0.7f + ((float) rand() / RAND_MAX) * 2.4f;
            node.depth_bias = ((float) rand() / RAND_MAX) * 0.55f;

            const float tint = (float) i / std::max(1, count - 1);
            node.color = {
                0.58f + 0.42f * tint,
                0.62f + 0.18f * (1.0f - tint),
                0.95f + 0.05f * std::sin(node.phase)
            };
            stars.push_back(node);
        }
    }

    static float clamp01(float v) {
        return std::clamp(v, 0.0f, 1.0f);
    }

    static uchar to_u8(float v) {
        return (uchar) std::lround(std::clamp(v, 0.0f, 255.0f));
    }

    cv::Vec3b make_color(float r, float g, float b, float intensity) const {
        return cv::Vec3b(
            to_u8(r * intensity * 255.0f),
            to_u8(g * intensity * 255.0f),
            to_u8(b * intensity * 255.0f)
        );
    }

    void add_pixel(cv::Mat& img, int x, int y, const cv::Vec3b& color, float alpha = 1.0f) const {
        if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) return;
        cv::Vec3b& dst = img.at<cv::Vec3b>(y, x);
        for (int c = 0; c < 3; ++c) {
            const float blended = dst[c] * (1.0f - alpha) + color[c] * alpha;
            dst[c] = to_u8(blended);
        }
    }

    void draw_star(cv::Mat& img, const ProjectedStar& star, float transient) const {
        const int radius = std::clamp((int)std::lround(star.size + transient * 2.4f), 1, 5);
        for (int oy = -radius; oy <= radius; ++oy) {
            for (int ox = -radius; ox <= radius; ++ox) {
                const float dist = std::sqrt((float)(ox * ox + oy * oy));
                if (dist > radius + 0.35f) continue;
                const float core = clamp01(1.0f - dist / (radius + 0.35f));
                const float glow = star.intensity * (0.45f + 0.55f * core);
                add_pixel(img,
                          (int)std::lround(star.pos.x) + ox,
                          (int)std::lround(star.pos.y) + oy,
                          make_color(star.color[0], star.color[1], star.color[2], glow),
                          clamp01(0.25f + core * 0.75f));
            }
        }
    }

    void draw_connection(cv::Mat& img, const ProjectedStar& a, const ProjectedStar& b, float intensity) const {
        const cv::Vec3b color = make_color(
            (a.color[0] + b.color[0]) * 0.5f,
            (a.color[1] + b.color[1]) * 0.5f,
            (a.color[2] + b.color[2]) * 0.5f,
            intensity
        );
        cv::line(img, a.pos, b.pos, cv::Scalar(color[0], color[1], color[2]), 1, cv::LINE_AA);
    }

    cv::Point2f swirl_point(const cv::Point2f& p, const AudioFeatures& f) const {
        const float cx = width * 0.5f;
        const float cy = height * 0.5f;
        const float dx = p.x - cx;
        const float dy = p.y - cy;
        const float radius = std::sqrt(dx * dx + dy * dy) + 1e-4f;

        const float angle = std::atan2(dy, dx) + std::sin(time_t * 0.55f + radius * 0.045f) * (0.06f + f.low * 0.33f);
        const float warp = 1.0f + std::sin(time_t * 0.9f + radius * 0.03f) * (0.01f + f.transient * 0.08f);
        return cv::Point2f(
            cx + std::cos(angle) * radius * warp,
            cy + std::sin(angle) * radius * warp
        );
    }

    ProjectedStar project_star(const StarNode& node, const AudioFeatures& f, float orbit_phase) const {
        const float breath = 0.10f + f.low * 0.40f;
        const float shimmer = 0.02f + f.high * 0.10f;
        const float pulse = std::sin(time_t * node.twinkle + node.phase);
        const float orbital = orbit_phase + node.phase * 0.12f;

        cv::Point3f pos = node.anchor;
        pos.x += std::cos(orbital + node.phase) * node.drift.x * breath;
        pos.y += std::sin(orbital * 0.8f + node.phase * 0.6f) * node.drift.y * (0.65f + f.mid * 0.7f);
        pos.z += std::sin(orbital * 1.1f + node.phase) * node.drift.z * (0.85f + f.low * 0.9f);

        pos.x += std::sin(time_t * 1.4f + node.phase * 2.0f) * shimmer;
        pos.y += std::cos(time_t * 1.7f + node.phase * 1.3f) * shimmer;
        pos.z += std::sin(time_t * 0.9f + node.phase * 0.8f) * (0.05f + f.transient * 0.18f);

        const float rot = time_t * (0.08f + f.rms * 0.35f);
        const float rx = pos.x * std::cos(rot) - pos.z * std::sin(rot);
        const float rz = pos.x * std::sin(rot) + pos.z * std::cos(rot);

        const float depth = rz + 2.4f + node.depth_bias;
        const float perspective = 0.78f / std::max(0.45f, depth);
        cv::Point2f screen(
            width * 0.5f + rx * perspective * width * 0.55f,
            height * 0.5f + pos.y * perspective * height * 0.70f
        );
        screen = swirl_point(screen, f);

        ProjectedStar projected;
        projected.pos = screen;
        projected.depth = depth;
        projected.intensity = clamp01(0.28f + (pulse * 0.5f + 0.5f) * 0.42f + f.high * 0.34f + f.transient * 0.25f);
        projected.size = node.size * (1.1f / std::max(0.75f, depth));
        projected.color = node.color;
        return projected;
    }

    void apply_pixel_distortion(cv::Mat& img, const AudioFeatures& f) const {
        const float amount = std::clamp(f.low * 0.55f + f.transient * 0.95f + f.high * 0.30f, 0.0f, 1.0f);
        if (amount < 0.04f) return;

        cv::Mat original = img.clone();
        const int band_step = std::max(2, 4 - (int)std::floor(f.high * 2.0f));

        for (int y = 0; y < img.rows; ++y) {
            const float wave = std::sin(time_t * 3.5f + y * 0.17f) * (1.0f + f.low * 5.0f);
            const int dx = (int)std::lround(wave * amount * 2.4f);
            if ((y / band_step) % 2 == 0) {
                for (int x = 0; x < img.cols; ++x) {
                    const int sx = std::clamp(x + dx, 0, img.cols - 1);
                    img.at<cv::Vec3b>(y, x) = original.at<cv::Vec3b>(y, sx);
                }
            }
        }

        const int center_y = std::clamp((int)std::lround(height * (0.5f + std::sin(time_t * 0.7f) * 0.22f)), 0, height - 1);
        const int smear_radius = std::max(2, (int)std::lround(2 + f.transient * 10.0f));
        for (int y = std::max(0, center_y - smear_radius); y <= std::min(height - 1, center_y + smear_radius); ++y) {
            const int shift = (int)std::lround(std::sin(time_t * 8.0f + y) * (1.0f + f.transient * 10.0f));
            for (int x = 0; x < width; ++x) {
                const int sx = std::clamp(x + shift, 0, width - 1);
                add_pixel(img, x, y, original.at<cv::Vec3b>(y, sx), 0.55f);
            }
        }
    }

    void soften_and_trail(cv::Mat& img, const AudioFeatures& f) {
        cv::Mat blurred;
        cv::GaussianBlur(img, blurred, cv::Size(0, 0), 0.8 + f.high * 1.4f);
        cv::addWeighted(img, 0.82f, blurred, 0.18f + f.high * 0.18f, 0.0, img);

        const float trail = std::clamp(0.10f + f.low * 0.20f + (1.0f - f.transient) * 0.12f, 0.08f, 0.34f);
        cv::addWeighted(img, 1.0f, prev_output, trail, 0.0, img);
        prev_output = img.clone();
    }

    cv::Mat update(const AudioFeatures& f) {
        time_t += 0.035f + f.rms * 0.018f;

        cv::Mat img(height, width, CV_8UC3, cv::Scalar(4, 2, 10));
        for (int y = 0; y < height; ++y) {
            const float vertical = (float) y / std::max(1, height - 1);
            const float haze = 0.10f + std::sin(time_t * 0.35f + vertical * 5.5f) * 0.04f + f.low * 0.12f;
            for (int x = 0; x < width; ++x) {
                cv::Vec3b& px = img.at<cv::Vec3b>(y, x);
                px[0] = to_u8(5.0f + haze * 34.0f);
                px[1] = to_u8(2.0f + haze * 18.0f);
                px[2] = to_u8(11.0f + haze * 52.0f);
            }
        }

        std::vector<ProjectedStar> projected;
        projected.reserve(stars.size());
        const float orbit_phase = time_t * (0.35f + f.low * 0.85f) + f.transient * 0.8f;

        for (const auto& star : stars) {
            ProjectedStar p = project_star(star, f, orbit_phase);
            if (p.pos.x < -8.0f || p.pos.x > width + 8.0f || p.pos.y < -8.0f || p.pos.y > height + 8.0f) continue;
            projected.push_back(p);
        }

        const float link_distance = 11.0f + f.mid * 24.0f + f.low * 10.0f;
        const float link_energy = std::clamp(f.mid * 0.85f + f.transient * 0.65f + f.high * 0.20f, 0.0f, 1.0f);
        for (size_t i = 0; i < projected.size(); ++i) {
            for (size_t j = i + 1; j < projected.size(); ++j) {
                const float dx = projected[i].pos.x - projected[j].pos.x;
                const float dy = projected[i].pos.y - projected[j].pos.y;
                const float dist = std::sqrt(dx * dx + dy * dy);
                if (dist > link_distance) continue;

                const float depth_diff = std::fabs(projected[i].depth - projected[j].depth);
                if (depth_diff > 0.65f + f.mid * 0.45f) continue;

                const float alpha = clamp01((1.0f - dist / link_distance) * (0.10f + link_energy * 0.95f));
                if (alpha > 0.03f) draw_connection(img, projected[i], projected[j], alpha);
            }
        }

        for (const auto& star : projected) {
            draw_star(img, star, f.transient);
        }

        apply_pixel_distortion(img, f);
        soften_and_trail(img, f);

        if (f.transient > 0.08f) {
            cv::Mat flash(height, width, CV_8UC3, cv::Scalar(
                to_u8(16.0f + f.high * 90.0f),
                to_u8(8.0f + f.mid * 40.0f),
                to_u8(14.0f + f.transient * 70.0f)
            ));
            cv::addWeighted(img, 1.0f, flash, std::clamp(f.transient * 0.12f, 0.0f, 0.20f), 0.0, img);
        }

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

    ConstellationEngine visual(LOGICAL_WIDTH, LOGICAL_HEIGHT);

    AudioAnalyzer audio;
    if (!audio.start()) {
        std::cerr << "Errore avvio audio analyzer.\n";
        return 1;
    }

    KeyboardInput keyboard;
    bool paused = false;
    int active_orientation_variant = 0;
    auto orientation_variant_until = std::chrono::steady_clock::time_point::min();

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
        auto now = std::chrono::steady_clock::now();
        if (paused) {
            features.rms *= 0.0f;
            features.low *= 0.0f;
            features.mid *= 0.0f;
            features.high *= 0.0f;
            features.transient *= 0.0f;
        }

        cv::Mat out = visual.update(features);

        const float orientation_energy = std::clamp(features.transient * 1.75f + features.high * 1.05f + features.rms * 0.60f, 0.0f, 1.0f);
        if (orientation_energy > 0.50f) {
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
