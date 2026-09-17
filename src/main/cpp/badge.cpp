// Puerto de piu_ocr/badge.py — tipo de chart por el color de la bolita.
#include "piu_ocr.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

namespace piu {

// Rangos de tono en la convención de OpenCV (H en 0..179). El naranja del
// single cruza el 0, así que va en dos tramos.
struct HueRange { const char* name; int lo1, hi1, lo2, hi2; };
static const std::array<HueRange, 4> kHues = {{
    {"single",     0,  22, 168, 179},   // naranja / rojo
    {"coop",      23,  34,  -1,  -1},   // amarillo
    {"double",    35,  85,  -1,  -1},   // verde
    {"halfdouble", 86, 130,  -1,  -1},  // azul / cian
}};

namespace {

// F3.7: cortes de S/V estimados del propio disco. Un disco lavado (S global
// bajo) ya no queda afuera del corte fijo 80/60. Se toma la mediana de S y un
// percentil bajo de V sobre los píxeles del disco (no sobre las esquinas de la
// caja, que son pantalla).
void adaptiveThresholds(const cv::Mat& s, const cv::Mat& v, const cv::Mat& disk,
                        float* sThr, float* vThr) {
  std::vector<float> sat, val;
  for (int y = 0; y < s.rows; ++y) {
    const uchar* sr = s.ptr<uchar>(y);
    const uchar* vr = v.ptr<uchar>(y);
    const uchar* dr = disk.ptr<uchar>(y);
    for (int x = 0; x < s.cols; ++x)
      if (dr[x]) { sat.push_back(float(sr[x])); val.push_back(float(vr[x])); }
  }
  if (sat.empty()) { *sThr = 80.f; *vThr = 60.f; return; }   // = umbral fijo legacy
  std::sort(sat.begin(), sat.end());
  std::sort(val.begin(), val.end());
  const float medS = sat[sat.size() / 2];
  const float v20 = val[size_t(0.20 * (val.size() - 1))];
  // El corte baja con el disco: si la mediana de S es 40 el corte es 20 (y no
  // 80, que borraba el disco entero). Pisos para no segmentar el ruido de fondo.
  *sThr = std::max(20.f, 0.5f * medS);
  *vThr = std::max(40.f, 0.5f * v20);
}

// F3.8: el sub-label (SINGLE/DOUBLE/HALFDOUBLE/COOP) que va sobre los dígitos
// del nivel dentro de la bolita. Se lee con chars.bin; si devuelve una palabra
// conocida con margen cómodo, el texto decide. Si no, manda el color.
const char* chartFromLabel(const std::string& t) {
  // "HALFDOUBLE" contiene "DOUBLE": hay que mirar primero la más específica.
  if (t.find("HALF") != std::string::npos) return "halfdouble";
  if (t.find("SINGLE") != std::string::npos) return "single";
  if (t.find("DOUBLE") != std::string::npos) return "double";
  if (t.find("COOP") != std::string::npos) return "coop";
  return nullptr;
}

}  // namespace

bool classifyChartType(const cv::Mat& roi, std::string* out, float* conf,
                       BadgeMode mode, const class Templates* chars) {
  if (roi.empty()) return false;
  cv::Mat hsv;
  cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> ch;
  cv::split(hsv, ch);

  const cv::Mat disk = discMask(hsv.rows, hsv.cols, DISC_R);
  cv::Mat mask;
  if (mode == BadgeMode::Color) {
    mask = (ch[1] > 80) & (ch[2] > 60);
  } else {
    float sThr = 80.f, vThr = 60.f;
    adaptiveThresholds(ch[1], ch[2], disk, &sThr, &vThr);
    mask = (ch[1] > sThr) & (ch[2] > vThr);
  }

  // La bolita es un disco inscrito en la caja, pero las esquinas de la caja son
  // pantalla, que en el layout Phoenix es azul. Con la caja floja de YOLO esas
  // esquinas ganaban el voto y 39 de 84 bolitas salían "halfdouble".
  // Restringir el voto al disco inscrito lo arregla (0.500 -> 0.833).
  mask &= disk;

  const int total = cv::countNonZero(mask);
  if (total < 20) return false;

  int bestCount = 0; const char* bestName = nullptr;
  for (const auto& r : kHues) {
    cv::Mat in = (ch[0] >= r.lo1) & (ch[0] <= r.hi1) & mask;
    int c = cv::countNonZero(in);
    if (r.lo2 >= 0) {
      cv::Mat in2 = (ch[0] >= r.lo2) & (ch[0] <= r.hi2) & mask;
      c += cv::countNonZero(in2);
    }
    if (c > bestCount) { bestCount = c; bestName = r.name; }
  }
  if (!bestName || bestCount == 0) return false;
  *out = bestName;
  *conf = float(bestCount) / float(total);

  // Fusión: el texto solo pisa al color si da una palabra conocida con margen
  // suficiente. Un sub-label ilegible (bajo ángulo/luz) no arruina el color.
  if (mode == BadgeMode::Fusion && chars && !chars->empty()) {
    auto gs = segmentChars(roi, 0, BinMode::Legacy);
    if (gs.size() >= 4) {
      std::vector<int> lab; std::vector<float> mar;
      chars->predict(gs, &lab, &mar);
      std::string txt;
      for (int L : lab)
        if (L > 0 && L < 128) txt += char(std::toupper(L));
      float m = 0.f;
      for (float v : mar) m += v;
      m = mar.empty() ? 0.f : m / mar.size();
      const char* t = chartFromLabel(txt);
      if (t && m > 0.05f) { *out = t; *conf = 0.95f; }
    }
  }
  return true;
}

}  // namespace piu
