// CLI de host: corre piu::Engine sobre una imagen y escribe JSON en stdout.
//
//   piuocr_cli --assets DIR IMG [--box cls,x1,y1,x2,y2,conf ...] [--augs 1,0.83,1f]
//
// --augs: pasadas del TTA, "escala" o "escalaf" (flip), separadas por coma.
// Sin --box corre el detector YOLO (con TTA, como el teléfono). Con --box usa
// esas cajas y saltea el detector, para aislar el OCR de la detección.
//
// Flags opt-in (F0.1): --rectify (F1), --bin-mode legacy|clahe|adaptive|all
// (F2), --badge-mode color|adaptive|fusion (F3), --title-variants N (F4).
// Defaults == comportamiento actual. Salida: {"result": ..., "boxes": [...],
// "ms": {...}}
#include "piu_ocr.h"
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace piu;

static double ms(std::chrono::steady_clock::time_point a) {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - a).count();
}

int main(int argc, char** argv) {
  std::string assets, image;
  std::vector<Box> given;
  bool useGiven = false;
  std::vector<Aug> augs;
  Options opts;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--assets") && i + 1 < argc) assets = argv[++i];
    else if (!std::strcmp(argv[i], "--rectify")) opts.rectify = true;
    else if (!std::strcmp(argv[i], "--bin-mode") && i + 1 < argc) {
      const std::string m = argv[++i];
      if (m == "clahe") opts.binMode = BinMode::Clahe;
      else if (m == "adaptive") opts.binMode = BinMode::Adaptive;
      else if (m == "all") opts.binMode = BinMode::All;
      else opts.binMode = BinMode::Legacy;
    } else if (!std::strcmp(argv[i], "--badge-mode") && i + 1 < argc) {
      const std::string m = argv[++i];
      if (m == "adaptive") opts.badgeMode = BadgeMode::Adaptive;
      else if (m == "fusion") opts.badgeMode = BadgeMode::Fusion;
      else opts.badgeMode = BadgeMode::Color;
    } else if (!std::strcmp(argv[i], "--title-variants") && i + 1 < argc) {
      opts.maxTitleVariants = std::max(1, std::atoi(argv[++i]));
    } else if (!std::strcmp(argv[i], "--title-boxes") && i + 1 < argc) {
      opts.maxTitleBoxes = std::max(1, std::atoi(argv[++i]));
    } else if (!std::strcmp(argv[i], "--box") && i + 1 < argc) {
      Box b{};
      if (std::sscanf(argv[++i], "%d,%d,%d,%d,%d,%f", &b.cls, &b.x1, &b.y1,
                      &b.x2, &b.y2, &b.conf) != 6) {
        std::fprintf(stderr, "--box mal formado: %s\n", argv[i]);
        return 2;
      }
      given.push_back(b);
      useGiven = true;
    } else if (!std::strcmp(argv[i], "--augs") && i + 1 < argc) {
      std::string spec = argv[++i];
      for (size_t p = 0; p < spec.size();) {
        size_t q = spec.find(',', p);
        if (q == std::string::npos) q = spec.size();
        std::string tok = spec.substr(p, q - p);
        Aug a{1.f, false};
        if (!tok.empty() && tok.back() == 'f') { a.flip = true; tok.pop_back(); }
        a.scale = std::strtof(tok.c_str(), nullptr);
        if (a.scale > 0.f) augs.push_back(a);
        p = q + 1;
      }
    } else if (argv[i][0] != '-') image = argv[i];
  }
  if (assets.empty() || image.empty()) {
    std::fprintf(stderr, "uso: piuocr_cli --assets DIR IMG [--box cls,x1,y1,x2,y2,conf ...]\n");
    return 2;
  }

  auto t0 = std::chrono::steady_clock::now();
  Engine eng;
  if (!eng.load(assets)) {
    std::fprintf(stderr, "no se pudieron cargar los assets de %s\n", assets.c_str());
    return 1;
  }
  const double loadMs = ms(t0);
  if (!augs.empty()) eng.augs = augs;
  eng.opts = opts;

  cv::Mat img = cv::imread(image, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::fprintf(stderr, "no se pudo leer %s\n", image.c_str());
    return 1;
  }

  std::vector<Box> used;
  t0 = std::chrono::steady_clock::now();
  const std::string result = eng.read(img, useGiven ? &given : nullptr, &used);
  const double readMs = ms(t0);

  std::printf("{\"result\":%s,\"boxes\":[", result.c_str());
  for (size_t i = 0; i < used.size(); ++i)
    std::printf("%s{\"cls\":%d,\"box\":[%d,%d,%d,%d],\"conf\":%g}", i ? "," : "",
                used[i].cls, used[i].x1, used[i].y1, used[i].x2, used[i].y2,
                used[i].conf);
  std::printf("],\"ms\":{\"load\":%.1f,\"read\":%.1f},\"detected\":%s}\n",
              loadMs, readMs, useGiven ? "false" : "true");
  return 0;
}
