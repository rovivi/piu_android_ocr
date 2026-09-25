// Detector YOLO sobre NCNN, con TTA implementado a mano.
//
// POR QUÉ EL TTA. Medido sobre 58 fotos de cabina, detecciones de song_name:
//   PyTorch con TTA     43 de 58
//   PyTorch sin TTA     29 de 58
//   NCNN    sin TTA     27 de 58
// El export a NCNN NO pierde nada (27 vs 29 es ruido); lo que se pierde es el
// TTA, que ultralytics aplica en PyTorch y NCNN ignora en silencio. Sin él, un
// tercio de los títulos no se detecta.
//
// Cuesta 3 pasadas. El detector ya es el 93 % del tiempo, así que esto es la
// decisión de latencia más cara del módulo — está tomada a conciencia.
#include "piu_ocr.h"
#include <cpu.h>
#include <net.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace piu {
namespace {

// Escalas del TTA. Las mismas que usa ultralytics para YOLO: la original más
// una reducción y un flip horizontal. Se pasó de 3 a 2 escalas porque la
// tercera no agregaba detecciones y sí un 33 % de costo.
//
// OJO: el export a NCNN es de FORMA FIJA (los Reshape del .param llevan
// 0=33600 anchors, los de 1280). Alimentar 1056 px devolvía basura con
// conf 0.9999 en la clase 1 que después ganaba el NMS: medido con el CLI de
// host, 0 de 58 song_name. Por eso la reducción se hace ADENTRO del lienzo de
// 1280 (la imagen ocupa el 83 % y el resto es relleno 114), que para una red
// convolucional es la misma augmentación sin cambiar la forma de entrada.
//
// Medido con tools/parity (--detect --augs), 45 fotos, recall de song_name y
// acierto de canción end-to-end:
//   1                 0.733  0.444   276 ms
//   1,1f              0.756  0.489   481
//   1,0.83,1f         0.844  0.644   661   <- lo que había
//   1,0.83f           0.933  0.733   486
//   1,0.83,0.83f      0.956  0.733   653   <- el que había: +2.3 pts de recall
//   1,0.83,1f,0.83f   0.933  0.733   844
// El flip sin reducir casi no suma; el flip DE la reducida sí.
//
// DOS PASADAS (2026-09-21). La tercera (0.83 sin flip) costaba un tercio del
// detector —que es ~90 % del read— por 2.3 pts de recall de song_name, y el
// acierto de canción end-to-end no cambiaba. Medido en device (Pixel 9 Pro XL,
// 62 fotos, decode de la app): el read completo promediaba 5.14 s y hasta
// 10.8 s, con 22 de 62 fotos pagando ADEMÁS la segunda detección del zoom. Esa
// es la cuenta que calienta el teléfono. En el emulador arm64, misma tanda:
//                precisión  cobertura  media
//   3 pasadas      0.886      0.646    958 ms
//   2 pasadas      0.889      0.667    629 ms   <- este
//   1 pasada       0.829      0.604    341 ms
// (la de 1 pasada pierde 7 campos de 192). Dos pasadas gana y pierde fotos
// sueltas contra tres (mismo total, ±1 campo), sin costo medible en precisión;
// la de una sola pasada es la que sí degrada. La segunda detección del zoomIn
// conserva estas mismas pasadas: bajarla a una perdió precisión (0.861).
const std::vector<Aug> kAugs = {{1.00f, false}, {0.83f, true}};

float iou(const Box& a, const Box& b) {
  const int x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
  const int x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
  const int w = std::max(0, x2 - x1), h = std::max(0, y2 - y1);
  const float inter = float(w) * h;
  const float ua = float(a.x2 - a.x1) * (a.y2 - a.y1)
                 + float(b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0.f;
}

// El .param trae la cantidad de anchors en los Reshape de la cabeza ("0=33600"). Con strides
// 8/16/32 un lienzo de lado s tiene s²·(1/64 + 1/256 + 1/1024) = s²·21/1024 anchors:
// 33600 → 1280, 21504 → 1024. Si no se encuentra, se usa `fallback`.
int imgszFromParam(const std::string& path, int fallback) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (!f) return fallback;
  char line[1024];
  int best = 0;
  while (std::fgets(line, sizeof line, f)) {
    if (std::strncmp(line, "Reshape", 7) != 0) continue;
    for (const char* p = std::strstr(line, " 0="); p; p = std::strstr(p + 1, " 0=")) {
      const int k = std::atoi(p + 3);
      const int s = int(std::lround(std::sqrt(double(k) * 1024.0 / 21.0)));
      if (s % 32 == 0 && (s / 8) * (s / 8) + (s / 16) * (s / 16) + (s / 32) * (s / 32) == k)
        best = std::max(best, s);
    }
  }
  std::fclose(f);
  return best > 0 ? best : fallback;
}

}  // namespace

const std::vector<Aug>& defaultAugs() { return kAugs; }

Detector::~Detector() { delete net_; }

bool Detector::load(const std::string& param, const std::string& bin) {
  delete net_;
  net_ = new ncnn::Net();
  net_->opt.use_vulkan_compute = false;   // arm64 CPU: predecible y sin drivers
  // Solo los cores grandes: un 4 fijo en un big.LITTLE 2+6 mete dos hilos en
  // cores chicos y el paso lo marca el más lento.
  ncnn::set_cpu_powersave(2);
  net_->opt.num_threads = std::max(1, std::min(4, ncnn::get_big_cpu_count()));
  net_->opt.lightmode = true;
  if (net_->load_param(param.c_str()) != 0 || net_->load_model(bin.c_str()) != 0) {
    delete net_;
    net_ = nullptr;
    return false;
  }
  imgsz_ = imgszFromParam(param, 1280);
  return true;
}

std::vector<Box> Detector::detectOnce(const cv::Mat& bgr, int imgsz,
                                      float scale, bool flip) const {
  // Letterbox al tamaño del modelo, igual que ultralytics.
  const int W = bgr.cols, H = bgr.rows;
  const int target = imgsz > 0 ? imgsz : imgsz_;  // forma fija, ver kAugs
  const float r = std::min(float(target) / W, float(target) / H) * scale;
  const int nw = int(std::round(W * r)), nh = int(std::round(H * r));
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(nw, nh));
  if (flip) cv::flip(resized, resized, 1);
  cv::Mat canvas(target, target, CV_8UC3, cv::Scalar(114, 114, 114));
  const int dx = (target - nw) / 2, dy = (target - nh) / 2;
  resized.copyTo(canvas(cv::Rect(dx, dy, nw, nh)));

  ncnn::Mat in = ncnn::Mat::from_pixels(canvas.data, ncnn::Mat::PIXEL_BGR2RGB,
                                        target, target);
  const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  in.substract_mean_normalize(nullptr, norm);

  ncnn::Extractor ex = net_->create_extractor();
  ex.input("in0", in);
  ncnn::Mat out;
  ex.extract("out0", out);
  if (std::getenv("PIU_DEBUG")) {       // solo en el CLI de host, nunca en Android
    std::fprintf(stderr, "out dims=%d w=%d h=%d c=%d target=%d r=%.4f\n",
                 out.dims, out.w, out.h, out.c, target, r);
    for (int y = 0; y < std::min(out.h, 9); ++y)
      std::fprintf(stderr, "  row%d: %.4f %.4f %.4f\n", y, out.row(y)[0],
                   out.row(y)[1], out.row(y)[out.w / 2]);
  }

  // out: (4 + nclases) x anchors
  std::vector<Box> boxes;
  const int nc = out.h - 4;
  for (int i = 0; i < out.w; ++i) {
    int best = 0; float bc = 0.f;
    for (int c = 0; c < nc; ++c) {
      const float v = out.row(4 + c)[i];
      if (v > bc) { bc = v; best = c; }
    }
    if (bc < 0.005f) continue;
    float cx = out.row(0)[i], cy = out.row(1)[i];
    const float w = out.row(2)[i], h = out.row(3)[i];
    if (flip) cx = target - cx;             // deshacer el flip
    Box b{int((cx - w / 2 - dx) / r), int((cy - h / 2 - dy) / r),
          int((cx + w / 2 - dx) / r), int((cy + h / 2 - dy) / r), bc};
    // Recortar a la imagen: el relleno del letterbox deja cajas negativas.
    b.x1 = std::max(0, std::min(W, b.x1)); b.x2 = std::max(0, std::min(W, b.x2));
    b.y1 = std::max(0, std::min(H, b.y1)); b.y2 = std::max(0, std::min(H, b.y2));
    if (b.x2 - b.x1 < 2 || b.y2 - b.y1 < 2) continue;
    b.cls = best;
    boxes.push_back(b);
  }
  return boxes;
}

std::vector<Box> Detector::detect(const cv::Mat& bgr, int imgsz,
                                  const std::vector<Aug>& augs) const {
  std::vector<Box> all;
  if (!net_ || bgr.empty()) return all;
  for (const Aug& a : augs) {
    auto v = detectOnce(bgr, imgsz, a.scale, a.flip);
    all.insert(all.end(), v.begin(), v.end());
  }
  // NMS por clase sobre la unión de las pasadas. Con umbral 0.005 las tres
  // pasadas juntan cientos de cajas: ordenar por (clase, conf) deja cada clase
  // contigua y el loop interno corta al cambiar de clase.
  std::sort(all.begin(), all.end(), [](const Box& a, const Box& b) {
    return a.cls != b.cls ? a.cls < b.cls : a.conf > b.conf;
  });
  std::vector<Box> keep;
  std::vector<bool> dead(all.size(), false);
  for (size_t i = 0; i < all.size(); ++i) {
    if (dead[i]) continue;
    keep.push_back(all[i]);
    for (size_t j = i + 1; j < all.size() && all[j].cls == all[i].cls; ++j)
      if (!dead[j] && iou(all[i], all[j]) > 0.45f) dead[j] = true;
  }
  std::sort(keep.begin(), keep.end(),
            [](const Box& a, const Box& b) { return a.conf > b.conf; });
  return keep;
}

}  // namespace piu
