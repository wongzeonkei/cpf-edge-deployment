#include <onnxruntime_cxx_api.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<float> read_bin_float(const std::string& path, size_t numel) {
    std::vector<float> data(numel);

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open input file: " + path);
    }

    ifs.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(numel * sizeof(float)));

    if (ifs.gcount() != static_cast<std::streamsize>(numel * sizeof(float))) {
        throw std::runtime_error("File size mismatch: " + path);
    }

    return data;
}

static void save_pgm_mask(const std::string& path, const float* logits, int h, int w, float threshold) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot open output file: " + path);
    }

    ofs << "P5\n" << w << " " << h << "\n255\n";

    for (int i = 0; i < h * w; ++i) {
        float prob = 1.0f / (1.0f + std::exp(-logits[i]));
        uint8_t value = prob >= threshold ? 255 : 0;
        ofs.write(reinterpret_cast<const char*>(&value), 1);
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0]
                  << " <model.onnx> <ct.bin> <pet.bin> <output_mask.pgm>\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string ct_path = argv[2];
    const std::string pet_path = argv[3];
    const std::string output_path = argv[4];

    constexpr int B = 1;
    constexpr int CT_C = 3;
    constexpr int PET_C = 1;
    constexpr int H = 512;
    constexpr int W = 512;
    constexpr float THRESHOLD = 0.35f;

    const size_t ct_numel = static_cast<size_t>(B) * CT_C * H * W;
    const size_t pet_numel = static_cast<size_t>(B) * PET_C * H * W;

    try {
        std::vector<float> ct = read_bin_float(ct_path, ct_numel);
        std::vector<float> pet = read_bin_float(pet_path, pet_numel);

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cpf_onnx_cpp");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session(env, model_path.c_str(), session_options);

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

        std::array<int64_t, 4> ct_shape = {B, CT_C, H, W};
        std::array<int64_t, 4> pet_shape = {B, PET_C, H, W};

        Ort::Value ct_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            ct.data(),
            ct.size(),
            ct_shape.data(),
            ct_shape.size()
        );

        Ort::Value pet_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            pet.data(),
            pet.size(),
            pet_shape.data(),
            pet_shape.size()
        );

        std::array<const char*, 2> input_names = {"ct", "pet"};
        std::array<const char*, 1> output_names = {"logits"};

        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(std::move(ct_tensor));
        input_tensors.emplace_back(std::move(pet_tensor));

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<Ort::Value> outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size()
        );

        auto end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

        float* logits = outputs[0].GetTensorMutableData<float>();

        save_pgm_mask(output_path, logits, H, W, THRESHOLD);

        std::cout << "ONNX Runtime C++ inference done.\n";
        std::cout << "Model: " << model_path << "\n";
        std::cout << "CT input: " << ct_path << "\n";
        std::cout << "PET input: " << pet_path << "\n";
        std::cout << "Output mask: " << output_path << "\n";
        std::cout << "Latency: " << latency_ms << " ms\n";

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
