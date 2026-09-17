// Puerto de piu_ocr/segment.py. Ver ../../../docs/ARQUITECTURA.md §7 antes de
// tocar cualquier constante: cada una salió de una medición.
#include "piu_ocr.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>

namespace piu {

float median(std::vector<float> v) {
  if (v.empty()) return 0.f;
  const size_t n = v.size(), h = n / 2;
  std::nth_element(v.begin(), v.begin() + h, v.end());
  const float hi = v[h];
  if (n % 2) return hi;
  const float lo = *std::max_element(v.begin(), v.begin() + h);
  return 0.5f * (lo + hi);
}

cv::Mat discMask(int H, int W, float r) {
  cv::Mat m(H, W, CV_8U);
  const double cy = (H - 1) / 2.0, cx = (W - 1) / 2.0;
  const double ry = std::max(H / 2.0, 1.0), rx = std::max(W / 2.0, 1.0);
  for (int y = 0; y < H; ++y) {
    uchar* row = m.ptr<uchar>(y);
    const double dy = (y - cy) / ry;
    for (int x = 0; x < W; ++x) {
      const double dx = (x - cx) / rx;
      row[x] = (dy * dy + dx * dx <= double(r) * r) ? 255 : 0;
    }
  }
  return m;
}

cv::Mat cropBox(const cv::Mat& img, const Box& b, float pad, float padX) {
  const float px = (padX < 0.f ? pad : padX);
  const int bw = b.x2 - b.x1, bh = b.y2 - b.y1;
  const int dx = int(bw * px), dy = int(bh * pad);
  const int x1 = std::max(0, b.x1 - dx), y1 = std::max(0, b.y1 - dy);
  const int x2 = std::min(img.cols, b.x2 + dx), y2 = std::min(img.rows, b.y2 + dy);
  if (x2 <= x1 || y2 <= y1) return cv::Mat();
  return img(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
}

// Normaliza un glifo a GLYPH_W x GLYPH_H conservando el aspecto.
static Glyph normGlyph(const cv::Mat& bin, const cv::Rect& r) {
  cv::Mat roi = bin(r);
  const float ar = float(r.width) / std::max(r.height, 1);
  const int tw = std::max(1, std::min(GLYPH_W, pyRound(double(GLYPH_H) * ar)));
  cv::Mat small;
  cv::resize(roi, small, cv::Size(tw, GLYPH_H), 0, 0, cv::INTER_AREA);
  cv::Mat out = cv::Mat::zeros(GLYPH_H, GLYPH_W, CV_8U);
  small.copyTo(out(cv::Rect((GLYPH_W - tw) / 2, 0, tw, GLYPH_H)));
  Glyph g;
  std::memcpy(g.px, out.data, sizeof(g.px));
  return g;
}

// Parte un componente lo bastante ancho como para ser dos dígitos pegados.
// En una foto de cabina los dos dígitos del nivel se funden: 56 de 82 bolitas
// segmentaban un solo glifo. El valle de la proyección vertical los separa.
static void splitMerged(const cv::Mat& m, std::vector<cv::Rect>* boxes) {
  std::vector<cv::Rect> out;
  for (const auto& r : *boxes) {
    if (r.width <= 0.95f * r.height || r.width < 6) { out.push_back(r); continue; }
    cv::Mat col;
    cv::reduce(m(r) > 0, col, 0, cv::REDUCE_SUM, CV_32S);
    const int lo = int(0.30f * r.width), hi = int(0.70f * r.width);
    if (hi <= lo) { out.push_back(r); continue; }
    int cut = lo; int best = col.at<int>(0, lo);
    for (int x = lo; x < hi; ++x)
      if (col.at<int>(0, x) < best) { best = col.at<int>(0, x); cut = x; }
    double mx; cv::minMaxLoc(col, nullptr, &mx);
    if (best > 0.55 * mx) { out.push_back(r); continue; }
    out.emplace_back(r.x, r.y, cut, r.height);
    out.emplace_back(r.x + cut, r.y, r.width - cut, r.height);
  }
  *boxes = out;
}

// Dobla la cantidad de componentes a una cantidad conocida. El nivel casi
// siempre son dos dígitos y el catálogo sabe cuántos, así que la cuenta es un
// dato de entrada, no una incógnita.
static void forceN(const cv::Mat& m, std::vector<cv::Rect>* boxes, int wantN) {
  if (boxes->empty()) return;
  auto byX = [](const cv::Rect& a, const cv::Rect& b) { return a.x < b.x; };
  std::sort(boxes->begin(), boxes->end(), byX);
  for (int guard = 0; int(boxes->size()) > wantN && guard < 20; ++guard) {
    size_t i = 0; int bestGap = INT32_MAX;
    for (size_t k = 0; k + 1 < boxes->size(); ++k) {
      const int gap = (*boxes)[k + 1].x - ((*boxes)[k].x + (*boxes)[k].width);
      if (gap < bestGap) { bestGap = gap; i = k; }
    }
    (*boxes)[i] |= (*boxes)[i + 1];          // unión de rectángulos
    boxes->erase(boxes->begin() + i + 1);
  }
  for (int guard = 0; int(boxes->size()) < wantN && guard < 20; ++guard) {
    size_t i = std::max_element(boxes->begin(), boxes->end(),
        [](const cv::Rect& a, const cv::Rect& b) { return a.width < b.width; })
        - boxes->begin();
    cv::Rect r = (*boxes)[i];
    if (r.width < 4) break;
    cv::Mat col;
    cv::reduce(m(r) > 0, col, 0, cv::REDUCE_SUM, CV_32S);
    const int lo = std::max(1, int(0.25f * r.width));
    const int hi = std::min(r.width - 1, int(0.75f * r.width));
    int cut = r.width / 2;
    if (hi > lo) {
      int best = col.at<int>(0, lo); cut = lo;
      for (int x = lo; x < hi; ++x)
        if (col.at<int>(0, x) < best) { best = col.at<int>(0, x); cut = x; }
    }
    boxes->erase(boxes->begin() + i);
    boxes->emplace_back(r.x, r.y, cut, r.height);
    boxes->emplace_back(r.x + cut, r.y, r.width - cut, r.height);
    std::sort(boxes->begin(), boxes->end(), byX);
  }
}

std::vector<Glyph> segmentBadge(const cv::Mat& roi, int wantN) {
  if (roi.empty()) return {};
  cv::Mat big, hsv;
  cv::resize(roi, big, cv::Size(), BADGE_SCALE, BADGE_SCALE, cv::INTER_CUBIC);
  cv::cvtColor(big, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> ch;
  cv::split(hsv, ch);
  cv::Mat m = (ch[1] < BADGE_S_MAX) & (ch[2] > BADGE_V_MIN);
  cv::morphologyEx(m, m, cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));

  // El borde brillante del disco también pasa el test de saturación baja y sale
  // como un arco que forceN parte en dos falsos dígitos. Filtrarlo por relleno
  // de la caja NO sirve (un `1` es tan hueco como un arco); enmascarar el disco
  // interior sí: los dígitos viven adentro, el borde por definición no.
  m &= discMask(m.rows, m.cols, BADGE_INNER_R);

  cv::Mat labels, stats, cent;
  const int n = cv::connectedComponentsWithStats(m, labels, stats, cent, 8);
  if (n <= 1) return {};
  int hmax = 0;
  for (int i = 1; i < n; ++i)
    hmax = std::max(hmax, stats.at<int>(i, cv::CC_STAT_HEIGHT));
  std::vector<cv::Rect> keep;
  for (int i = 1; i < n; ++i) {
    const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
    const int a = stats.at<int>(i, cv::CC_STAT_AREA);
    // La palabra SINGLE/DOUBLE va en letra mucho más chica arriba del número.
    if (h < 0.55f * hmax || a < 0.02f * m.total()) continue;
    keep.emplace_back(stats.at<int>(i, cv::CC_STAT_LEFT),
                      stats.at<int>(i, cv::CC_STAT_TOP),
                      stats.at<int>(i, cv::CC_STAT_WIDTH), h);
  }
  std::sort(keep.begin(), keep.end(),
            [](const cv::Rect& a, const cv::Rect& b) { return a.x < b.x; });
  splitMerged(m, &keep);
  if (wantN > 0) forceN(m, &keep, wantN);
  std::vector<Glyph> out;
  out.reserve(keep.size());
  for (const auto& r : keep) out.push_back(normGlyph(m, r));
  return out;
}

}  // namespace piu
