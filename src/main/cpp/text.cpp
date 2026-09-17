// Puerto de piu_ocr/text.py — lectura del título de la canción.
#include "piu_ocr.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <iterator>
#include <numeric>

namespace piu {

// Funde componentes que se solapan horizontalmente: el punto de la i y la
// diéresis son componentes aparte del mismo carácter.
static std::vector<cv::Rect> mergeByX(std::vector<cv::Rect> b, float overlap = 0.55f) {
  if (b.empty()) return b;
  std::sort(b.begin(), b.end(), [](const cv::Rect& a, const cv::Rect& c) { return a.x < c.x; });
  std::vector<cv::Rect> out{b[0]};
  for (size_t i = 1; i < b.size(); ++i) {
    cv::Rect& p = out.back();
    const int inter = std::min(p.x + p.width, b[i].x + b[i].width) - std::max(p.x, b[i].x);
    if (inter > 0 && inter >= overlap * std::min(p.width, b[i].width)) p |= b[i];
    else out.push_back(b[i]);
  }
  return out;
}

// Se queda con la corrida de glifos que forma UNA línea de texto. Un recorte
// ancho no trae solo el título: trae el BPM, el "FREE PLAY" y el nombre del
// jugador, todos pasan los filtros de ruido y ensucian la cadena. Vale 15
// puntos de acierto, y además hace que el lector no dependa de que la caja de
// YOLO esté bien ajustada a lo ancho.
std::vector<cv::Rect> dominantLine(std::vector<cv::Rect> b) {
  if (b.size() < 3) return b;
  std::sort(b.begin(), b.end(), [](const cv::Rect& a, const cv::Rect& c) { return a.x < c.x; });
  std::vector<float> hs;
  for (const auto& r : b) hs.push_back(float(r.height));
  const float ref = median(hs);
  std::vector<cv::Rect> keep;
  for (const auto& r : b)
    if (std::abs(r.height - ref) <= DOM_H_TOL * ref) keep.push_back(r);
  if (keep.size() < 3) return b;

  std::vector<int> gaps;
  for (size_t i = 0; i + 1 < keep.size(); ++i)
    gaps.push_back(keep[i + 1].x - (keep[i].x + keep[i].width));
  std::vector<float> pos;
  for (int g : gaps) if (g >= 0) pos.push_back(float(g));
  const float med = median(pos);
  const float limit = std::max(med * DOM_GAP_MULT, 0.9f * ref);

  std::vector<std::vector<cv::Rect>> runs{{keep[0]}};
  for (size_t i = 0; i < gaps.size(); ++i) {
    if (gaps[i] > limit) runs.push_back({});
    runs.back().push_back(keep[i + 1]);
  }
  return *std::max_element(runs.begin(), runs.end(),
      [](const std::vector<cv::Rect>& a, const std::vector<cv::Rect>& c) {
        int sa = 0, sc = 0;
        for (auto& r : a) sa += r.width;
        for (auto& r : c) sc += r.width;
        return sa < sc;
      });
}

// Reencuentra la banda del título dentro de un recorte flojo: es la corrida de
// filas más brillante. Las filas de afuera son portada del álbum, que es lo que
// envenena los umbrales.
cv::Mat focusBand(const cv::Mat& roi) {
  if (roi.empty() || roi.rows < 12) return roi;
  cv::Mat g;
  if (roi.channels() == 3) cv::cvtColor(roi, g, cv::COLOR_BGR2GRAY); else g = roi;
  cv::Mat prof;
  cv::reduce(g, prof, 1, cv::REDUCE_AVG, CV_32F);
  cv::GaussianBlur(prof, prof, {1, 5}, 0);
  double lo, hi;
  cv::minMaxLoc(prof, &lo, &hi);
  if (hi - lo < 12) return roi;
  const float thr = float(lo + 0.55 * (hi - lo));
  int bestLen = 0, bestY0 = 0, bestY1 = 0, cur = -1;
  for (int y = 0; y <= g.rows; ++y) {
    const bool on = y < g.rows && prof.at<float>(y, 0) >= thr;
    if (on && cur < 0) cur = y;
    else if (!on && cur >= 0) {
      if (y - cur > bestLen) { bestLen = y - cur; bestY0 = cur; bestY1 = y; }
      cur = -1;
    }
  }
  if (bestLen < 0.25f * g.rows) return roi;
  const int pad = std::max(2, int(0.18f * bestLen));
  const int y0 = std::max(0, bestY0 - pad), y1 = std::min(g.rows, bestY1 + pad);
  return roi(cv::Rect(0, y0, roi.cols, y1 - y0));
}

// Imágenes a umbralizar. El título va sobre una banda con degradado, así que un
// umbral global se come la mitad de las letras; el top-hat lo saca.
// OJO: MORPH_RECT sería separable y ~14 ms más barato, pero es medible peor
// (0.733 vs 0.800): las esquinas del rectángulo llegan más lejos en diagonal y
// la banda cruza el recorte en diagonal, así que come trazo además de fondo.
//
// F2: CLAHE y adaptiveThreshold atacan luz irregular/glare. En Legacy NO se
// agregan: el default tiene que dar byte-a-byte lo mismo que antes. En All se
// corren todos y el mismo puntaje de coherencia elige; Legacy va primero para
// que en empate gane el camino viejo.
static void variants(const cv::Mat& gray, BinMode mode, std::vector<cv::Mat>* out) {
  out->push_back(gray);
  const int k = std::max(3, (pyRound(0.55 * gray.rows) | 1));
  cv::Mat th;
  cv::morphologyEx(gray, th, cv::MORPH_TOPHAT,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, {k, k}));
  out->push_back(th);
  if (mode == BinMode::Legacy) return;
  if (mode == BinMode::Clahe || mode == BinMode::All) {
    // equalizeHist local: realza el contraste por bloques sin mover el
    // histograma global, que es lo que rompe el percentil en luz despareja.
    cv::Mat eq;
    cv::Ptr<cv::CLAHE> cl = cv::createCLAHE(2.0, {8, 8});
    cl->apply(gray, eq);
    out->push_back(eq);
  }
  if (mode == BinMode::Adaptive || mode == BinMode::All) {
    cv::Mat ad;
    const int bs = std::max(11, (gray.rows / 8) | 1);
    cv::adaptiveThreshold(gray, ad, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, bs, 8);
    out->push_back(ad);
  }
}

// F1.3.5: deskew fino del título. La pendiente dominante de la banda se estima
// con Hough sobre los bordes; si está dentro de ±maxDeg se corrige con
// warpAffine. Identidad si no hay líneas claras o si la pendiente es despreciable.
cv::Mat deskewBand(const cv::Mat& roi, float maxDeg) {
  if (roi.empty() || roi.rows < 12 || roi.cols < 12) return roi;
  cv::Mat gray;
  if (roi.channels() == 3) cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY); else gray = roi;
  cv::Mat edges;
  cv::Canny(gray, edges, 50, 150);
  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(edges, lines, 1.0, CV_PI / 180.0,
                  std::max(10, gray.cols / 8), gray.cols / 4.0, 8);
  std::vector<std::pair<float, float>> ang;  // (ángulo en grados, peso=largo)
  for (const cv::Vec4i& l : lines) {
    const float dx = float(l[2] - l[0]), dy = float(l[3] - l[1]);
    const float len = std::hypot(dx, dy);
    float a = float(std::atan2(dy, dx) * 180.0 / CV_PI);
    while (a >= 90.f) a -= 180.f;
    while (a < -90.f) a += 180.f;
    if (std::abs(a) <= maxDeg) ang.emplace_back(a, len);
  }
  if (ang.empty()) return roi;
  std::sort(ang.begin(), ang.end(),
            [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
              return a.first < b.first;
            });
  float total = 0.f;
  for (const auto& p : ang) total += p.second;
  float acc = 0.f, med = 0.f;
  for (const auto& p : ang) {
    acc += p.second;
    if (acc >= 0.5f * total) { med = p.first; break; }
  }
  if (std::abs(med) < 0.3f) return roi;
  const cv::Point2f c(roi.cols / 2.f, roi.rows / 2.f);
  // getRotationMatrix2D con ángulo positivo gira en sentido antihorario: una
  // línea que baja a la derecha (ángulo +a) vuelve a horizontal con +a.
  const cv::Mat M = cv::getRotationMatrix2D(c, med, 1.0);
  cv::Mat out;
  cv::warpAffine(roi, out, M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
  return out;
}

// Percentil sobre un histograma de 256 bins: O(n) una sola vez por fuente en
// vez de copiar la imagen entera a un vector y hacer nth_element 6 veces.
// Mismo resultado que nth_element sobre el índice p/100*(n-1).
static void histogram(const cv::Mat& m, int (&h)[256]) {
  std::fill(std::begin(h), std::end(h), 0);
  for (int y = 0; y < m.rows; ++y) {
    const uchar* row = m.ptr<uchar>(y);
    for (int x = 0; x < m.cols; ++x) ++h[row[x]];
  }
}

static float percentile(const int (&h)[256], size_t n, float p) {
  if (n == 0) return 0.f;
  const size_t target = std::min(n - 1, size_t(p / 100.f * (n - 1)));
  size_t acc = 0;
  for (int v = 0; v < 256; ++v) {
    acc += size_t(h[v]);
    if (acc > target) return float(v);
  }
  return 255.f;
}

static Glyph normInLine(const cv::Mat& bin, const cv::Rect& r, int y0, int lineH) {
  cv::Mat roi = bin(r);
  const int gh = std::max(1, pyRound(double(GLYPH_H) * r.height / lineH));
  const int gw = std::max(1, std::min(GLYPH_W,
      pyRound(double(gh) * r.width / std::max(r.height, 1))));
  cv::Mat small;
  cv::resize(roi, small, cv::Size(gw, gh), 0, 0, cv::INTER_AREA);
  cv::Mat out = cv::Mat::zeros(GLYPH_H, GLYPH_W, CV_8U);
  int top = pyRound(double(GLYPH_H) * (r.y - y0) / lineH);
  top = std::max(0, std::min(GLYPH_H - gh, top));
  small.copyTo(out(cv::Rect((GLYPH_W - gw) / 2, top, gw, gh)));
  Glyph gl;
  std::memcpy(gl.px, out.data, sizeof(gl.px));
  return gl;
}

namespace {
// Una lectura completa de la caja: el conjunto de glifos y su puntaje. Se
// materializan los glifos al vuelo (son chicos) para no guardar una imagen
// umbralizada por candidato.
struct Reading {
  float scoreA = -1e9f, scoreB = -1e9f;
  std::vector<Glyph> glyphs;
};

bool sameReading(const std::vector<Glyph>& a, const std::vector<Glyph>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(a[i].px, b[i].px, sizeof(a[i].px))) return false;
  return true;
}

void collectReadings(const cv::Mat& gray, BinMode mode, int wantN,
                     std::vector<Reading>* out) {
  std::vector<cv::Mat> srcs;
  variants(gray, mode, &srcs);
  for (const cv::Mat& src : srcs) {
    int hist[256];
    histogram(src, hist);
    for (int p : PCTS) {
      cv::Mat bin = src > percentile(hist, src.total(), float(p));
      cv::Mat labels, stats, cent;
      const int n = cv::connectedComponentsWithStats(bin, labels, stats, cent, 8);
      if (n <= 1) continue;
      int amax = 0;
      for (int i = 1; i < n; ++i) amax = std::max(amax, stats.at<int>(i, cv::CC_STAT_AREA));
      std::vector<cv::Rect> cand;
      for (int i = 1; i < n; ++i) {
        const int a = stats.at<int>(i, cv::CC_STAT_AREA);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        if (a < 0.02f * amax || h < 0.12f * gray.rows) continue;
        if (w > 0.5f * gray.cols) continue;   // banda de fondo, no una letra
        cand.emplace_back(stats.at<int>(i, cv::CC_STAT_LEFT),
                          stats.at<int>(i, cv::CC_STAT_TOP), w, h);
      }
      cand = mergeByX(cand);
      cand = dominantLine(cand);
      if (cand.empty()) continue;
      float mean = 0.f;
      for (auto& r : cand) mean += r.height;
      mean /= cand.size();
      float var = 0.f;
      for (auto& r : cand) var += (r.height - mean) * (r.height - mean);
      const float coh = -std::sqrt(var / cand.size()) / std::max(mean, 1e-6f);
      const float a = (wantN > 0) ? -std::abs(int(cand.size()) - wantN)
                                  : float(cand.size());
      int y0 = INT32_MAX, y1 = 0;
      for (const auto& r : cand) { y0 = std::min(y0, r.y); y1 = std::max(y1, r.y + r.height); }
      const int lineH = std::max(1, y1 - y0);
      Reading rd;
      rd.scoreA = a; rd.scoreB = coh;
      rd.glyphs.reserve(cand.size());
      for (const auto& r : cand) rd.glyphs.push_back(normInLine(bin, r, y0, lineH));
      out->push_back(std::move(rd));
    }
  }
  std::stable_sort(out->begin(), out->end(),
                   [](const Reading& a, const Reading& b) {
                     return a.scoreA != b.scoreA ? a.scoreA > b.scoreA
                                                 : a.scoreB > b.scoreB;
                   });
}
}  // namespace

std::vector<std::vector<Glyph>> segmentCharsVariants(const cv::Mat& roi, int wantN,
                                                     BinMode mode) {
  std::vector<std::vector<Glyph>> out;
  if (roi.empty()) return out;
  cv::Mat gray;
  if (roi.channels() == 3) cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY); else gray = roi;

  // Normalizar la altura de trabajo en vez de escalar a ciegas. Un recorte de
  // una foto de 2048 px daba una imagen de 4500x900 umbralizada 12 veces: una
  // sola tardó 106 SEGUNDOS. Además sube la precisión, porque el kernel del
  // top-hat y los umbrales de ruido pasan a significar lo mismo en una foto y
  // en un frame de video.
  float f = float(TARGET_H) / std::max(gray.rows, 1);
  f = std::min(std::max(f, MIN_SCALE), 3.f);
  if (std::abs(f - 1.f) > 0.02f)
    cv::resize(gray, gray, cv::Size(), f, f, f > 1 ? cv::INTER_CUBIC : cv::INTER_AREA);

  std::vector<Reading> reads;
  collectReadings(gray, mode, wantN, &reads);
  for (const Reading& r : reads) {
    if (r.glyphs.empty()) continue;
    bool dup = false;
    for (const auto& g : out)
      if (sameReading(g, r.glyphs)) { dup = true; break; }
    if (dup) continue;
    out.push_back(r.glyphs);
    if (out.size() >= 2) break;   // solo interesan la mejor y la 2ª
  }
  return out;
}

std::vector<Glyph> segmentChars(const cv::Mat& roi, int wantN, BinMode mode) {
  auto v = segmentCharsVariants(roi, wantN, mode);
  return v.empty() ? std::vector<Glyph>{} : v.front();
}

}  // namespace piu
