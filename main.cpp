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


using rgb_matrix::RGBMatrix;
using rgb_matrix::Canvas;
using rgb_matrix::FrameCanvas;

struct AudioFeatures {
    float rms = 0.0f;
    float low = 0.0f;
    float mid = 0.0f;
    float high = 0.0f;
    float transient = 0.0f;
};

#if defined(__linux__) && defined(SUPPRESS_ALSA_WARNINGS)
static void silent_alsa_error_handler(const char*, int, const char*, int, const char*, ...) {}
#endif

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

        for (unsigned long i = 0; i < framesPerBuffer; i++) {
            float l = input[i * input_channels];
            float r = (input_channels > 1) ? input[i * input_channels + 1] : l;
            mono[i] = 0.5f * (l + r);
        }

        // RMS
        float rms = 0.0f;
        for (auto s : mono) rms += s * s;
        rms = std::sqrt(rms / mono.size());

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

        // smoothing
        smooth_rms  = smooth_value(smooth_rms,  raw_rms,  0.25f, 0.05f);
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

class VisualEngine {
public:
    int width, height;
    float time_t = 0.0f;

    cv::Mat base_img;      // RGB
    cv::Mat prev_luma;     // float
    cv::Mat luma;          // float
    cv::Mat edge_map;      // float
    cv::Mat motion_map;    // float

    VisualEngine(const cv::Mat& initial_frame, int w, int h)
        : width(w), height(h)
    {
        base_img = initial_frame.clone();
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

    cv::Mat pixel_sort(const cv::Mat& img, float amount, float threshold_bias = 0.0f) {
        if (amount < 0.03f) return img.clone();

        cv::Mat out = img.clone();
        cv::Mat luma8;
        cv::cvtColor(img, luma8, cv::COLOR_RGB2GRAY);

        const int max_span = std::max(6, (int)(amount * 24.0f));
        const int min_run = std::max(3, (int)(3 + amount * 8.0f));
        const int threshold = std::clamp((int)(110 + threshold_bias * 60.0f + amount * 50.0f), 32, 220);
        const bool vertical_pass = amount > 0.30f;

        auto sort_segment = [&](int fixed, int start, int end, bool vertical) {
            if (end - start < min_run) return;

            std::vector<std::pair<int, cv::Vec3b>> segment;
            segment.reserve(end - start);

            for (int pos = start; pos < end; ++pos) {
                cv::Vec3b px = vertical ? out.at<cv::Vec3b>(pos, fixed) : out.at<cv::Vec3b>(fixed, pos);
                int score = px[0] * 77 + px[1] * 150 + px[2] * 29;
                segment.push_back({score, px});
            }

            std::stable_sort(segment.begin(), segment.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
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
                int strength = luma8.at<uchar>(y, x) + (int)(motion_map.at<float>(y, x) * 255.0f);
                if (strength > threshold) {
                    int start = x;
                    while (x < img.cols && x - start < max_span) {
                        int current = luma8.at<uchar>(y, x) + (int)(motion_map.at<float>(y, x) * 255.0f);
                        if (current <= threshold) break;
                        ++x;
                    }
                    sort_segment(y, start, x, false);
                } else {
                    ++x;
                }
            }
        }

        if (!vertical_pass) return out;

        for (int x = 0; x < img.cols; ++x) {
            int y = 0;
            while (y < img.rows) {
                int strength = luma8.at<uchar>(y, x) + (int)(edge_map.at<float>(y, x) * 255.0f);
                if (strength > threshold + 10) {
                    int start = y;
                    while (y < img.rows && y - start < max_span / 2) {
                        int current = luma8.at<uchar>(y, x) + (int)(edge_map.at<float>(y, x) * 255.0f);
                        if (current <= threshold + 10) break;
                        ++y;
                    }
                    sort_segment(x, start, y, true);
                } else {
                    ++y;
                }
            }
        }

        return out;
    }

    cv::Mat update(const AudioFeatures& f) {
        time_t += 0.05f;

        cv::Mat img = base_img.clone();
        img = apply_red_grade(img);

        img = background_mass_displacement(img, f.low * 0.9f + f.mid * 0.5f);
        img = contour_displacement_static_only(img, f.mid * 1.4f + f.low * 0.4f);
        img = edge_rgb_glitch_static_only(img, f.high * 1.3f + f.mid * 0.5f);
        img = edge_color_burn_static_only(img, f.high * 1.1f + f.transient * 0.8f);
        img = pixel_sort(img, std::clamp(f.mid * 0.9f + f.high * 0.7f + f.transient * 0.6f, 0.0f, 1.0f), f.low - 0.25f);

        img = preserve_moving_areas(img);
        img = preserve_stillness(img, f.rms);

        return img;
    }
};

// --------------------------------------------
// Utils
// --------------------------------------------
bool is_black_frame(const cv::Mat& frame, int threshold = 18, float dark_ratio = 0.92f) {
    cv::Mat luma;
    cv::cvtColor(frame, luma, cv::COLOR_RGB2GRAY);

    int dark = 0;
    int total = luma.rows * luma.cols;

    for (int y = 0; y < luma.rows; y++) {
        for (int x = 0; x < luma.cols; x++) {
            if (luma.at<uchar>(y, x) < threshold) dark++;
        }
    }

    return ((float)dark / total) > dark_ratio;
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

cv::Mat remap_for_panel_layout(const cv::Mat& frame) {
    const int panel_w = frame.cols / 2;
    const int panel_h = frame.rows / 2;

    cv::Mat out(frame.rows, frame.cols, frame.type());

    cv::Rect src_p1(panel_w, 0, panel_w, panel_h);
    cv::Rect src_p2(0, panel_h, panel_w, panel_h);
    cv::Rect src_p3(0, 0, panel_w, panel_h);
    cv::Rect src_p4(panel_w, panel_h, panel_w, panel_h);

    apply_panel_transform(frame(src_p3), 270, false, false).copyTo(out(cv::Rect(0, 0, panel_w, panel_h)));
    apply_panel_transform(frame(src_p1), 270, false, false).copyTo(out(cv::Rect(panel_w, 0, panel_w, panel_h)));
    apply_panel_transform(frame(src_p2), 90, false, false).copyTo(out(cv::Rect(0, panel_h, panel_w, panel_h)));
    apply_panel_transform(frame(src_p4), 90, false, false).copyTo(out(cv::Rect(panel_w, panel_h, panel_w, panel_h)));

    return out;
}

// --------------------------------------------
// MAIN
// --------------------------------------------
int main(int argc, char *argv[]) {
    srand(time(nullptr));

    // In rpi-rgb-led-matrix rows/cols sono la misura del singolo pannello.
    // Per un 128x128 composto da 4 pannelli 64x64 il layout corretto e' 64x64, chain=2, parallel=2.
    const int PANEL_ROWS = env_to_int("MATRIX_ROWS", 64);
    const int PANEL_COLS = env_to_int("MATRIX_COLS", 64);
    const int CHAIN_LENGTH = env_to_int("MATRIX_CHAIN", 2);
    const int PARALLEL = env_to_int("MATRIX_PARALLEL", 2);
    const int WIDTH = PANEL_COLS * CHAIN_LENGTH;
    const int HEIGHT = PANEL_ROWS * PARALLEL;
    const std::string VIDEO_PATH = "video.mov";

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

    cv::VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) {
        std::cerr << "Impossibile aprire video: " << VIDEO_PATH << "\n";
        return 1;
    }

    std::vector<cv::Mat> random_buffer = preload_random_frames(VIDEO_PATH, WIDTH, HEIGHT, 100);
    if (random_buffer.empty()) {
        std::cerr << "Buffer random vuoto.\n";
    }

    cv::Mat frame;
    cap.read(frame);
    cv::resize(frame, frame, cv::Size(WIDTH, HEIGHT));
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

    VisualEngine visual(frame, WIDTH, HEIGHT);

    AudioAnalyzer audio;
    if (!audio.start()) {
        std::cerr << "Errore avvio audio analyzer.\n";
        return 1;
    }

    bool kick_triggered = false;
    auto last_kick = std::chrono::steady_clock::now();

    cv::Mat frozen_output;
    bool has_frozen_frame = false;

    // soglia sotto la quale consideriamo "silenzio"
    const float SILENCE_THRESHOLD = 0.015f;
    auto last_audio_time = std::chrono::steady_clock::now();
    const float SILENCE_HOLD_SEC = 0.35f;

    while (true) {
        // --------------------------------
        // QUI devi collegare il tuo audio vero
        // --------------------------------
        AudioFeatures features = audio.getFeatures();
        if (features.rms >= SILENCE_THRESHOLD) {
            last_audio_time = std::chrono::steady_clock::now();
        }

        float silence_elapsed = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - last_audio_time
        ).count();

        bool no_audio = silence_elapsed > SILENCE_HOLD_SEC;

        if (no_audio && has_frozen_frame) {
            // Se non c'e' audio, tieni l'ultimo frame processato fermo.
            draw_to_matrix(offscreen, frozen_output);
            offscreen = matrix->SwapOnVSync(offscreen);
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            continue;
        }
        // --------------------------------
        // KICK JUMP con cooldown
        // --------------------------------
        auto now = std::chrono::steady_clock::now();
        float since_kick = std::chrono::duration<float>(now - last_kick).count();

        if (features.transient > 0.28f && features.low > 0.45f && !kick_triggered && since_kick > 0.35f) {
            if (!random_buffer.empty()) {
                int idx = rand() % random_buffer.size();
                visual.base_img = random_buffer[idx].clone();
                visual.luma = visual.compute_luma(visual.base_img);
                visual.edge_map = visual.compute_edge_map(visual.luma);
                visual.motion_map = visual.compute_motion_map(visual.luma);
            }

            kick_triggered = true;
            last_kick = now;
        }
        else if (features.low <= 0.32f) {
            kick_triggered = false;
        }

        // --------------------------------
        // Playback normale
        // --------------------------------
        if (!cap.read(frame)) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            cap.read(frame);
        }

        cv::resize(frame, frame, cv::Size(WIDTH, HEIGHT));
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        if (is_black_frame(frame) && !random_buffer.empty()) {
            int idx = rand() % random_buffer.size();
            frame = random_buffer[idx].clone();
        }

        visual.base_img = frame.clone();
        visual.luma = visual.compute_luma(visual.base_img);
        visual.edge_map = visual.compute_edge_map(visual.luma);
        visual.motion_map = visual.compute_motion_map(visual.luma);

        cv::Mat out = visual.update(features);
        out = remap_for_panel_layout(out);
        frozen_output = out.clone();
        has_frozen_frame = true;

        draw_to_matrix(offscreen, out);
        offscreen = matrix->SwapOnVSync(offscreen);

        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60fps target
    }

    audio.stop();
    delete matrix;
    return 0;
}
