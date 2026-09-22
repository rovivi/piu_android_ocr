// Puerto de segment_digits (piu_ocr/segment.py) + read_score (recognize.py).
//
// El score es la única franja de la pantalla que es una sola línea de dígitos
// de altura pareja, así que no necesita el top-hat ni la línea dominante del
// título: alcanza con barrer umbrales y quedarse con el conjunto de glifos más
// plausible. Lo que sí necesita, y el título no, es preguntarle al
// clasificador: la regularidad sola no distingue un "8" de un borrón relleno
// por un umbral malo — los dos son un blob del tamaño correcto.
#include "piu_ocr.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace piu {
namespace {

// np.percentile con interpolación lineal (el default de numpy), sobre un
// histograma de 256 bins. Tomar el elemento de orden a secas da un uint8 y
// mueve el umbral hasta un nivel de gris entero, que en un recorte de score
// chico cambia qué componentes sobreviven.
float percentileLinear(const int (&h)[256], size_t n, float p) {
  if (n == 0) return 0.f;
  const double idx = double(p) / 100.0 * (n - 1);
  const size_t lo = size_t(std::floor(idx));
  const double frac = idx - double(lo);
  size_t acc = 0;
  int vlo = 255, vhi = 255;
  bool haveLo = false;
  for (int v = 0; v < 256; ++v) {
    acc += size_t(h[v]);
    if (!haveLo && acc > lo) { vlo = v; haveLo = true; }
    if (acc > lo + 1) { vhi = v; break; }
  }
  if (!haveLo) return 255.f;
  return float(vlo + frac * (vhi - vlo));
}

struct Cand { cv::Rect r; float bright; };

// _norm_glyph: recorta y normaliza a GLYPH_W x GLYPH_H conservando el aspecto.
// A diferencia del título, acá el glifo ocupa TODO el alto: en una línea de
// dígitos no hay minúsculas ni ascendentes que distinguir.
Glyph normGlyph(const cv::Mat& bin, const cv::Rect& r) {
  const float ar = float(r.width) / std::max(r.height, 1);
  const int tw = std::max(1, std::min(GLYPH_W, pyRound(double(GLYPH_H) * ar)));
  cv::Mat small;
  cv::resize(bin(r), small, cv::Size(tw, GLYPH_H), 0, 0, cv::INTER_AREA);
  cv::Mat out = cv::Mat::zeros(GLYPH_H, GLYPH_W, CV_8U);
  small.copyTo(out(cv::Rect((GLYPH_W - tw) / 2, 0, tw, GLYPH_H)));
  Glyph g;
  std::memcpy(g.px, out.data, sizeof(g.px));
  return g;
}

// _same_line: se queda con los glifos de la línea del score. El recorte
// arrastra el bonus ("+84,818") que va debajo y en cuerpo más chico; los
// dígitos del score comparten altura y línea base.
std::vector<cv::Rect> sameLine(std::vector<cv::Rect> boxes) {
  if (boxes.size() <= 1) return boxes;
  int hmax = 0;
  for (const auto& b : boxes) hmax = std::max(hmax, b.height);
  std::vector<cv::Rect> band;
  for (const auto& b : boxes)
    if (b.height >= 0.68f * hmax && b.height <= 1.45f * hmax) band.push_back(b);
  if (band.empty()) return boxes;

  std::vector<float> cys, hs;
  for (const auto& b : band) {
    cys.push_back(b.y + b.height / 2.0f);
    hs.push_back(float(b.height));
  }
  // OJO: cy es cys[n/2] del vector ORDENADO (el de arriba del medio si n es
  // par), no la mediana de numpy; href sí es la mediana de numpy. Están así en
  // segment.py y la diferencia se nota con 6 dígitos.
  std::sort(cys.begin(), cys.end());
  const float cy = cys[cys.size() / 2];
  const float href = median(hs);

  std::vector<cv::Rect> keep;
  for (const auto& b : band)
    if (std::abs((b.y + b.height / 2.0f) - cy) <= 0.42f * href) keep.push_back(b);
  return keep;
}

struct Extract {
  std::vector<cv::Rect> boxes;
  std::vector<Glyph> glyphs;
};

// _extract: componentes conexas -> glifos filtrados por altura, relleno,
// proporción y brillo.
Extract extract(const cv::Mat& th, const cv::Mat& gray) {
  Extract out;
  cv::Mat labels, stats, cent;
  const int n = cv::connectedComponentsWithStats(th, labels, stats, cent, 8);
  if (n <= 1) return out;

  float hmax = 0.f;
  for (int i = 1; i < n; ++i)
    hmax = std::max(hmax, float(stats.at<int>(i, cv::CC_STAT_HEIGHT)));

  // Brillo medio de cada componente en una sola pasada: PIU dibuja los ceros a
  // la izquierda no significativos en gris apagado y Otsu los deja pasar de
  // forma inconsistente, así que se filtran por brillo, no por posición.
  std::vector<double> sum(n, 0.0);
  std::vector<int> cnt(n, 0);
  for (int y = 0; y < labels.rows; ++y) {
    const int* lrow = labels.ptr<int>(y);
    const uchar* grow = gray.ptr<uchar>(y);
    for (int x = 0; x < labels.cols; ++x) {
      const int L = lrow[x];
      if (L > 0) { sum[L] += grow[x]; ++cnt[L]; }
    }
  }

  std::vector<Cand> cand;
  for (int i = 1; i < n; ++i) {
    const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
    const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
    const int area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (h < 0.55f * hmax) continue;          // "SCORE", el "+84,818", ruido
    if (area < 0.05f * w * h) continue;      // rayas finas y bordes
    if (w > 3.0f * h) continue;              // barras horizontales
    if (cnt[i] == 0) continue;
    cand.push_back({cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT),
                             stats.at<int>(i, cv::CC_STAT_TOP), w, h),
                    float(sum[i] / cnt[i])});
  }
  if (cand.empty()) return out;

  float bmax = 0.f;
  for (const auto& c : cand) bmax = std::max(bmax, c.bright);
  std::vector<cv::Rect> keep;
  for (const auto& c : cand)
    if (c.bright >= 0.62f * bmax) keep.push_back(c.r);

  keep = sameLine(keep);
  std::sort(keep.begin(), keep.end(),
            [](const cv::Rect& a, const cv::Rect& b) { return a.x < b.x; });
  out.boxes = keep;
  out.glyphs.reserve(keep.size());
  for (const auto& r : keep) out.glyphs.push_back(normGlyph(th, r));
  return out;
}

// _plausibility: cuenta esperada + regularidad de altura y de paso.
std::pair<int, float> plausibility(const std::vector<cv::Rect>& boxes) {
  const int n = int(boxes.size());
  if (n == 0) return {-1, 0.f};
  const int countScore = (n >= 6 && n <= 7) ? 2 : ((n >= 4 && n <= 8) ? 1 : 0);

  double mean = 0.0;
  for (const auto& b : boxes) mean += b.height;
  mean /= n;
  double var = 0.0;
  for (const auto& b : boxes) var += (b.height - mean) * (b.height - mean);
  float reg = -float(std::sqrt(var / n) / std::max(mean, 1e-6));

  if (n >= 2) {
    // El paso se mide entre CENTROS, no entre bordes izquierdos: el "1" es más
    // angosto que el resto y midiendo por borde el conjunto que lo descarta
    // parece más regular, con lo cual "1000000" se leía como "0".
    std::vector<double> xs;
    for (const auto& b : boxes) xs.push_back(b.x + b.width / 2.0);
    std::sort(xs.begin(), xs.end());
    std::vector<double> gaps;
    for (size_t i = 1; i < xs.size(); ++i) gaps.push_back(xs[i] - xs[i - 1]);
    double gm = 0.0;
    for (double g : gaps) gm += g;
    gm /= gaps.size();
    double gv = 0.0;
    for (double g : gaps) gv += (g - gm) * (g - gm);
    reg -= float(std::sqrt(gv / gaps.size()) / std::max(gm, 1e-6));
  }
  return {countScore, reg};
}

}  // namespace

// F2: agrega a la lista de umbralizados una versión con CLAHE y una
// adaptiveThreshold. Legacy no agrega nada (el default no cambia). En All, los
// viejos van primero, así que en empate gana el camino de siempre.
static void addLightVariants(const cv::Mat& gray, BinMode mode,
                             std::vector<cv::Mat>* cands) {
  if (mode == BinMode::Legacy) return;
  if (mode == BinMode::Clahe || mode == BinMode::All) {
    cv::Mat eq;
    cv::Ptr<cv::CLAHE> cl = cv::createCLAHE(2.0, {8, 8});
    cl->apply(gray, eq);
    cv::Mat otsu;
    cv::threshold(eq, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cands->push_back(otsu);
    cv::Mat inv;
    cv::bitwise_not(otsu, inv);
    cands->push_back(inv);
    int hist[256];
    std::fill(std::begin(hist), std::end(hist), 0);
    for (int y = 0; y < eq.rows; ++y) {
      const uchar* row = eq.ptr<uchar>(y);
      for (int x = 0; x < eq.cols; ++x) ++hist[row[x]];
    }
    for (int p : {60, 70, 78, 85, 90, 94})
      cands->push_back(eq > percentileLinear(hist, eq.total(), float(p)));
  }
  if (mode == BinMode::Adaptive || mode == BinMode::All) {
    cv::Mat ad;
    const int bs = std::max(11, (gray.rows / 2) | 1);
    cv::adaptiveThreshold(gray, ad, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, bs, 5);
    cands->push_back(ad);
    cv::Mat adi;
    cv::bitwise_not(ad, adi);
    cands->push_back(adi);
  }
}

std::vector<Glyph> segmentDigits(const cv::Mat& roi, const Templates& digits,
                                 int scale, BinMode mode) {
  if (roi.empty()) return {};
  cv::Mat gray;
  if (roi.channels() == 3) cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY); else gray = roi;
  if (scale != 1) {
    // 4x sobre una caja de primer plano (miles de px) multiplicaba cada
    // umbralizado intermedio hasta pedir GB y matar el proceso. La caja más
    // grande medida en device (1409x730 → 16.5 Mpx escalados) no se toca; el
    // tope solo recorta el caso patológico, donde el glifo ya es enorme.
    // ponytail: tope de área; si el dataset crece a 4K, medir contra 20e6.
    constexpr double MAX_SCALED_AREA = 20e6;
    const double area = double(gray.rows) * double(gray.cols);
    int s = scale;
    if (area > 0 && area * double(s) * s > MAX_SCALED_AREA)
      s = std::max(1, int(std::sqrt(MAX_SCALED_AREA / area)));
    if (s != 1)
      cv::resize(gray, gray, cv::Size(), s, s, cv::INTER_CUBIC);
  }

  // Un umbral global único falla en pantallas de fondo claro: se barren varios
  // y gana el que produce el conjunto de glifos más plausible. Otsu y su
  // inverso van primero porque son los que aciertan casi siempre.
  std::vector<cv::Mat> cands;
  cv::Mat otsu;
  cv::threshold(gray, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  cands.push_back(otsu);
  cv::Mat inv;
  cv::bitwise_not(otsu, inv);
  cands.push_back(inv);

  int hist[256];
  std::fill(std::begin(hist), std::end(hist), 0);
  for (int y = 0; y < gray.rows; ++y) {
    const uchar* row = gray.ptr<uchar>(y);
    for (int x = 0; x < gray.cols; ++x) ++hist[row[x]];
  }
  for (int p : {60, 70, 78, 85, 90, 94})
    cands.push_back(gray > percentileLinear(hist, gray.total(), float(p)));

  addLightVariants(gray, mode, &cands);

  std::vector<Glyph> best;
  int bestCount = -2;
  float bestReg = 0.f;
  for (const cv::Mat& th : cands) {
    Extract e = extract(th, gray);
    auto key = plausibility(e.boxes);
    if (key.first > 0 && !e.glyphs.empty()) {
      // La regularidad sola no distingue un "8" real de un borrón relleno por
      // un umbral malo: los dos son un blob del tamaño correcto. Se le pregunta
      // al clasificador cuán reconocibles son los glifos.
      std::vector<int> lab; std::vector<float> mar;
      digits.predict(e.glyphs, &lab, &mar);
      float m = 0.f;
      for (float v : mar) m += v;
      key.second += 3.0f * (mar.empty() ? -1.f : m / mar.size());
    }
    if (key.first > bestCount || (key.first == bestCount && key.second > bestReg)) {
      bestCount = key.first; bestReg = key.second; best = e.glyphs;
    }
    // Salida temprana: con la cuenta correcta y el clasificador cómodo no vale
    // la pena barrer el resto de los umbrales.
    if (bestCount == 2 && bestReg > 0.10f) break;
  }
  return best;
}

bool readScore(const cv::Mat& roi, const Templates& digits, int* value,
               float* margin, std::string* rawDigits, BinMode mode) {
  *value = -1;
  *margin = 0.f;
  rawDigits->clear();
  auto gs = segmentDigits(roi, digits, 4, mode);
  if (gs.empty()) return false;

  std::vector<int> lab; std::vector<float> mar;
  digits.predict(gs, &lab, &mar);
  std::string s;
  for (int L : lab) {
    const int d = digitOf(L);
    if (d < 0) return false;
    s += char('0' + d);
  }
  *rawDigits = s;
  *margin = mar.empty() ? 0.f : *std::min_element(mar.begin(), mar.end());

  // Restricción del dominio: el score de PIU Phoenix va de 0 a 1 000 000. El
  // único valor legítimo de 7 dígitos es 1000000; cualquier otro de 7 dígitos
  // significa que el cero gris de la izquierda se coló como glifo.
  if (s.size() == 7 && s != "1000000") s = s.substr(1);
  if (s.empty() || s.size() > 7) return false;
  long v = std::stol(s);
  if (v > 1000000) return false;
  *value = int(v);
  return true;
}

}  // namespace piu
