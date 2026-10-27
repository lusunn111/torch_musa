// Standalone public-API reproducer. Does not link to PyTorch or use model data.
#include <mudnn.h>
#include <musa_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define MU(call)                                                           \
  do {                                                                     \
    auto status = (call);                                                  \
    if (status != musaSuccess) {                                           \
      std::fprintf(stderr, "%s: %s\n", #call, musaGetErrorString(status)); \
      return 2;                                                            \
    }                                                                      \
  } while (0)
#define DNN(call)                                                  \
  do {                                                             \
    auto status = (call);                                          \
    if (status != MUDNN_STATUS_SUCCESS) {                          \
      std::fprintf(stderr, "%s: status %d\n", #call, int(status)); \
      return 2;                                                    \
    }                                                              \
  } while (0)

int main(int argc, char** argv) {
  int rows = 8128, n = 537, repeats = 3000, pad = 0;
  float mask = -10000.f;
  for (int arg = 1; arg < argc; ++arg) {
    if (!std::strcmp(argv[arg], "--rows"))
      rows = std::atoi(argv[++arg]);
    else if (!std::strcmp(argv[arg], "--columns"))
      n = std::atoi(argv[++arg]);
    else if (!std::strcmp(argv[arg], "--repeats"))
      repeats = std::atoi(argv[++arg]);
    else if (!std::strcmp(argv[arg], "--pad"))
      pad = std::atoi(argv[++arg]);
    else if (!std::strcmp(argv[arg], "--mask"))
      mask = std::strtof(argv[++arg], nullptr);
    else
      return 2;
  }
  if (rows <= 0 || n <= 1 || repeats <= 0 || pad < 0)
    return 2;
  int stride = pad ? ((n + pad - 1) / pad) * pad : n;
  const int active = std::min(256, n / 2);
  std::vector<float> input(size_t(rows) * stride, mask), output(input.size());
  std::vector<double> expected(input.size());
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < active; ++c)
      input[size_t(r) * stride + c] =
          .5f * std::sin(float(r % 97) * .07f + float(c) * .03f);
    input[size_t(r) * stride] = 1.f;
    double sum = 0;
    for (int c = 0; c < stride; ++c)
      sum += std::exp(double(input[size_t(r) * stride + c]) - 1.);
    for (int c = 0; c < stride; ++c)
      expected[size_t(r) * stride + c] =
          std::exp(double(input[size_t(r) * stride + c]) - 1.) / sum;
  }
  MU(musaSetDevice(0));
  float *x = nullptr, *y = nullptr;
  MU(musaMalloc(reinterpret_cast<void**>(&x), input.size() * sizeof(float)));
  MU(musaMalloc(reinterpret_cast<void**>(&y), output.size() * sizeof(float)));
  MU(musaMemcpy(
      x, input.data(), input.size() * sizeof(float), musaMemcpyHostToDevice));
  mudnnHandle_t handle;
  mudnnTensorDescriptor_t xd, yd;
  mudnnSoftmaxDescriptor_t op;
  DNN(mudnnCreate(&handle));
  DNN(mudnnSetStream(handle, nullptr));
  DNN(mudnnCreateTensorDescriptor(&xd));
  DNN(mudnnCreateTensorDescriptor(&yd));
  int dims[2] = {rows, stride}, strides[2] = {stride, 1};
  DNN(mudnnSetTensorNdDescriptor(xd, MUDNN_DATA_FLOAT, 2, dims, strides));
  DNN(mudnnSetTensorNdDescriptor(yd, MUDNN_DATA_FLOAT, 2, dims, strides));
  DNN(mudnnCreateSoftmaxDescriptor(&op));
  DNN(mudnnSetSoftmaxDescriptor(
      op, 1, MUDNN_SOFTMAX_ACCURATE, MUDNN_SOFTMAX_MODE_SOFTMAX));
  size_t bytes = 0;
  void* workspace = nullptr;
  DNN(mudnnGetSoftmaxForwardWorkspaceSize(handle, op, xd, yd, &bytes));
  if (bytes)
    MU(musaMalloc(&workspace, bytes));
  std::printf(
      "rows=%d columns=%d stored_columns=%d mask=%g workspace=%zu\n",
      rows,
      n,
      stride,
      mask,
      bytes);
  int failures = 0;
  for (int iteration = 0; iteration < repeats; ++iteration) {
    DNN(mudnnSoftmaxForwardEx(handle, op, xd, x, workspace, bytes, yd, y));
    MU(musaMemcpy(
        output.data(),
        y,
        output.size() * sizeof(float),
        musaMemcpyDeviceToHost));
    bool failed = false;
    for (int r = 0; r < rows; ++r) {
      double sum = 0;
      for (int c = 0; c < stride; ++c) {
        size_t i = size_t(r) * stride + c;
        sum += output[i];
        if (!std::isfinite(output[i]) ||
            std::abs(double(output[i]) - expected[i]) > 1e-5) {
          if (!failed)
            std::printf(
                "FAIL iteration=%d row=%d col=%d input=%g actual=%g expected=%.9g\n",
                iteration,
                r,
                c,
                input[i],
                output[i],
                expected[i]);
          failed = true;
        }
      }
      if (!std::isfinite(sum) || std::abs(sum - 1.) > 1e-5)
        failed = true;
    }
    failures += failed;
    if ((iteration + 1) % 500 == 0) {
      std::printf("iterations=%d failures=%d\n", iteration + 1, failures);
      std::fflush(stdout);
    }
  }
  std::printf("RESULT iterations=%d failures=%d\n", repeats, failures);
  DNN(mudnnDestroySoftmaxDescriptor(op));
  DNN(mudnnDestroyTensorDescriptor(xd));
  DNN(mudnnDestroyTensorDescriptor(yd));
  DNN(mudnnDestroy(handle));
  if (workspace)
    MU(musaFree(workspace));
  MU(musaFree(x));
  MU(musaFree(y));
  return failures ? 1 : 0;
}
