// Orquestación de los campos que viven en C++. Sin JNI a propósito: el .so y
// el CLI de host (tools/host/cli.cpp) llaman exactamente a esto, así que lo
// que mide el test de paridad es lo que corre el teléfono.
#include "piu_ocr.h"
#include <algorithm>
#include <sstream>
#include <string>

namespace piu {
namespace {

// JSON válido siempre: además de " y \, escapa controles y todo lo que no sea
// ASCII como \uXXXX. Las etiquetas de chars.bin son códigos de carácter y un
// byte suelto >= 0x80 rompería el parser de JSONObject.
std::string esc(const std::string& s) {
  static const char* hex = "0123456789abcdef";
  std::string o;
  o.reserve(s.size() + 8);
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += char(c); }
    else if (c >= 0x20 && c < 0x7f) o += char(c);
    else { o += "\\u00"; o += hex[c >> 4]; o += hex[c & 15]; }
  }
  return o;
}

// Caja en píxeles de la imagen leída, recortada a ella. Las cajas viajan en
// el JSON para DIBUJAR sobre la foto (la app anima dónde vio cada cosa);
// ningún gate las lee.
void putBox(std::ostringstream& js, const Box& b, int W, int H) {
  auto cl = [](int v, int hi) { return std::max(0, std::min(v, hi)); };
  js << "[" << cl(b.x1, W) << "," << cl(b.y1, H) << "," << cl(b.x2, W) << ","
     << cl(b.y2, H) << "]";
}

}  // namespace

const char* Engine::emptyJson() {
  return "{\"titles\":[],\"score\":-1,\"score_margin\":0,\"score_digits\":\"\","
         "\"chart_type\":\"\",\"chart_conf\":0,"
         "\"level_digits\":[],\"level_scores\":[]}";
}

bool Engine::load(const std::string& base) {
  chars_ = Templates::load(base + "/chars.bin");
  level_ = Templates::load(base + "/level.bin");
  digits_ = Templates::load(base + "/digits.bin");
  return !chars_.empty() && !level_.empty() && !digits_.empty() &&
         det_.load(base + "/piu_yolo.param", base + "/piu_yolo.bin");
}

namespace {
// Por debajo de esta fracción del encuadre la pantalla está LEJOS: en el
// lienzo de 1280 el título mide diez píxeles y el detector no lo ve.
constexpr float ZOOM_MAX_SHARE = 0.40f;
// Margen alrededor de la pantalla al recortarla: el título asoma del borde
// en algunas fotos y una caja al ras lo partía.
constexpr float ZOOM_PAD = 0.06f;
constexpr int   ZOOM_MIN_SIDE = 64;
}  // namespace

// El zoom. En una foto lejana la pantalla es una franja del encuadre; el
// detector la mira en un lienzo de 1280 y pierde el título (song_name es lo
// más chico que busca). Si marcó la pantalla entera (fullscore, recall 1.0) y
// ocupa poco de la foto, se vuelve a detectar SOLO dentro de ella: para la red
// es la misma pantalla llenando el lienzo. Las cajas vuelven en coordenadas de
// la foto completa y los recortes del OCR siguen saliendo de la imagen
// original, así que el resto de la cadena no se entera.
std::vector<Box> Engine::zoomIn(const cv::Mat& img, std::vector<Box> first) const {
  const Box* panel = nullptr;
  for (const Box& b : first)
    if (b.cls == 1 && (!panel || b.conf > panel->conf)) panel = &b;
  if (!panel) return first;
  const int W = img.cols, H = img.rows;
  const int pw = panel->x2 - panel->x1, ph = panel->y2 - panel->y1;
  if (pw <= 0 || ph <= 0 || W <= 0 || H <= 0) return first;
  if (float(pw) * float(ph) >= ZOOM_MAX_SHARE * float(W) * float(H)) return first;
  const int x1 = std::max(0, panel->x1 - int(pw * ZOOM_PAD));
  const int y1 = std::max(0, panel->y1 - int(ph * ZOOM_PAD));
  const int x2 = std::min(W, panel->x2 + int(pw * ZOOM_PAD));
  const int y2 = std::min(H, panel->y2 + int(ph * ZOOM_PAD));
  if (x2 - x1 < ZOOM_MIN_SIDE || y2 - y1 < ZOOM_MIN_SIDE) return first;

  cv::Mat crop = img(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
  std::vector<Box> second = det_.detect(crop, 1280, augs);
  for (Box& b : second) { b.x1 += x1; b.x2 += x1; b.y1 += y1; b.y2 += y1; }

  // Lo que la segunda pasada vio manda; lo que no vio se conserva de la primera.
  bool has[5] = {false, false, false, false, false};
  for (const Box& b : second) if (b.cls >= 0 && b.cls < 5) has[b.cls] = true;
  for (const Box& b : first)
    if (b.cls >= 0 && b.cls < 5 && !has[b.cls]) second.push_back(b);
  return second;
}

std::string Engine::read(const cv::Mat& img, const std::vector<Box>* given,
                         std::vector<Box>* usedBoxes) const {
  // imgsz 1280 con TTA: es la única configuración que da cajas usables. Bajar a
  // 768 detecta MÁS cajas pero peor puestas, y el OCR consume el recorte —
  // medido, cuesta canción 0.800 -> 0.633.
  // PRIMER PASO: zoom. Ver zoomIn.
  const std::vector<Box> boxes = given ? *given : zoomIn(img, det_.detect(img, 1280, augs));
  if (usedBoxes) *usedBoxes = boxes;

  std::ostringstream js;
  js.imbue(std::locale::classic());     // "0.5", nunca "0,5"
  js << "{\"titles\":[";
  std::string chartType;
  float chartConf = 0.f;
  std::vector<int> lvlDigits;
  std::vector<std::vector<float>> lvlScores;

  // song_name: hasta 3 cajas, de mayor a menor confianza.
  std::vector<Box> titles;
  for (const Box& b : boxes) if (b.cls == 4) titles.push_back(b);
  std::stable_sort(titles.begin(), titles.end(),
                   [](const Box& a, const Box& b) { return a.conf > b.conf; });
  if (titles.size() > 3) titles.resize(3);
  bool first = true;
  for (const Box& b : titles) {
    cv::Mat roi = cropBox(img, b, 0.04f);
    if (roi.empty()) continue;
    roi = focusBand(roi);
    auto gs = segmentChars(roi);
    if (gs.empty()) continue;
    std::vector<int> lab; std::vector<float> mar;
    chars_.predict(gs, &lab, &mar);
    std::string txt;
    for (int L : lab) txt += char(L);
    if (!first) js << ",";
    js << "{\"raw\":\"" << esc(txt) << "\",\"conf\":" << b.conf << ",\"box\":";
    putBox(js, b, img.cols, img.rows);
    js << "}";
    first = false;
  }

  // difficulty: SOLO la bolita de mayor confianza, como _first() en
  // pipeline.py. Tomar el mejor voto entre todas las cajas parecía más
  // robusto y era peor (paridad: 0.881 -> 0.810): una segunda caja floja
  // sobre pantalla azul vota "halfdouble" con conf 1.0 y le gana a la real.
  const Box* diff = nullptr;
  for (const Box& b : boxes)
    if (b.cls == 0 && (!diff || b.conf > diff->conf)) diff = &b;
  if (diff) {
    // pad +0.02: medido sobre 55 bolitas, -0.02 da 0.855 y +0.02 da 0.873.
    std::string t; float cf = 0.f;
    if (classifyChartType(cropBox(img, *diff, 0.02f), &t, &cf)) {
      chartType = t; chartConf = cf;
    }
    auto gs = segmentBadge(cropBox(img, *diff, -0.02f), 2);
    if (!gs.empty()) {
      std::vector<float> mar;
      level_.predict(gs, &lvlDigits, &mar);
      level_.scores(gs, &lvlScores);
      for (int& d : lvlDigits) d = digitOf(d);   // code point -> dígito
    }
  }

  // score: la caja de mayor confianza, como _first() en pipeline.py. Devuelve
  // el margen MÍNIMO entre los dígitos; el gate vive en Kotlin.
  const Box* scoreBox = nullptr;
  for (const Box& b : boxes)
    if (b.cls == 3 && (!scoreBox || b.conf > scoreBox->conf)) scoreBox = &b;
  int scoreVal = -1;
  float scoreMargin = 0.f;
  std::string scoreDigits;
  if (scoreBox)
    readScore(cropBox(img, *scoreBox, 0.06f), digits_, &scoreVal, &scoreMargin,
              &scoreDigits);

  // rank: no se lee, pero es una de las cosas que la pantalla muestra y la
  // app la señala al animar la lectura.
  const Box* rankBox = nullptr;
  for (const Box& b : boxes)
    if (b.cls == 2 && (!rankBox || b.conf > rankBox->conf)) rankBox = &b;
  // fullscore: la pantalla de resultado entera. Es adonde zoomIn acercó la
  // detección y adonde la app acerca la foto antes de mostrar lo leído.
  const Box* screenBox = nullptr;
  for (const Box& b : boxes)
    if (b.cls == 1 && (!screenBox || b.conf > screenBox->conf)) screenBox = &b;

  js << "],\"w\":" << img.cols << ",\"h\":" << img.rows;
  js << ",\"screen_box\":";
  if (screenBox) putBox(js, *screenBox, img.cols, img.rows); else js << "null";
  js << ",\"score_box\":";
  if (scoreBox) putBox(js, *scoreBox, img.cols, img.rows); else js << "null";
  js << ",\"badge_box\":";
  if (diff) putBox(js, *diff, img.cols, img.rows); else js << "null";
  js << ",\"rank_box\":";
  if (rankBox) putBox(js, *rankBox, img.cols, img.rows); else js << "null";

  js << ",\"score\":" << scoreVal << ",\"score_margin\":" << scoreMargin
     << ",\"score_digits\":\"" << esc(scoreDigits) << "\"";
  js << ",\"chart_type\":\"" << esc(chartType) << "\",\"chart_conf\":"
     << chartConf << ",\"level_digits\":[";
  for (size_t i = 0; i < lvlDigits.size(); ++i) js << (i ? "," : "") << lvlDigits[i];
  js << "],\"level_scores\":[";
  for (size_t i = 0; i < lvlScores.size(); ++i) {
    js << (i ? ",[" : "[");
    for (size_t j = 0; j < lvlScores[i].size(); ++j)
      js << (j ? "," : "") << lvlScores[i][j];
    js << "]";
  }
  js << "]}";
  return js.str();
}

}  // namespace piu
