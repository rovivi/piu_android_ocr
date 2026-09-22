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

// F4.10: tras el rectify, un título que quedó por debajo de ~24 px en la
// pantalla enderezada ya llega degradado: segmentChars normaliza a TARGET_H,
// pero la interpolación no inventa trazo. Se hace una detección local alrededor
// de esa caja (el título pasa a medir cientos de px para la red) y se reemplaza
// si la nueva caja es más alta. Opt-in: solo se llama en el camino rectify.
std::vector<Box> Engine::zoomTinyTitles(const cv::Mat& work,
                                        std::vector<Box> boxes) const {
  constexpr int MIN_TITLE_PX = 24;
  const int W = work.cols, H = work.rows;
  for (Box& b : boxes) {
    if (b.cls != 4) continue;
    const int bw = b.x2 - b.x1, bh = b.y2 - b.y1;
    if (bh >= MIN_TITLE_PX || bw < 8 || bh < 4) continue;
    const int x1 = std::max(0, b.x1 - int(bw * 0.6f));
    const int y1 = std::max(0, b.y1 - int(bh * 1.0f));
    const int x2 = std::min(W, b.x2 + int(bw * 0.6f));
    const int y2 = std::min(H, b.y2 + int(bh * 1.0f));
    if (x2 - x1 < 20 || y2 - y1 < 12) continue;
    cv::Mat crop = work(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    std::vector<Box> sec = det_.detect(crop, 1280, augs);
    const Box* best = nullptr;
    for (const Box& s : sec)
      if (s.cls == 4 && (!best || s.conf > best->conf)) best = &s;
    if (!best) continue;
    Box nb = *best;
    nb.x1 += x1; nb.x2 += x1; nb.y1 += y1; nb.y2 += y1;
    if (nb.y2 - nb.y1 > bh) b = nb;
  }
  return boxes;
}

namespace {
// ¿El detector marcó la pantalla entera (fullscore)? Es la señal de que la
// pantalla se ve derecha; si falta, la foto puede estar girada 90°.
bool hasScreen(const std::vector<Box>& boxes) {
  for (const Box& b : boxes)
    if (b.cls == 1) return true;
  return false;
}

// Cuánto ve una orientación: la pantalla (fullscore) manda — vale 100 — y si
// no, pesan las clases presentes y la confianza de la mejor caja de cada una.
float orientationScore(const std::vector<Box>& boxes) {
  float top[5] = {0, 0, 0, 0, 0};
  for (const Box& b : boxes)
    if (b.cls >= 0 && b.cls < 5 && b.conf > top[b.cls]) top[b.cls] = b.conf;
  float s = 0.f;
  for (float c : top) if (c > 0) s += 1.f + c;
  return top[1] > 0 ? 100.f + s : s;
}
}  // namespace

std::string Engine::read(const cv::Mat& img, const std::vector<Box>* given,
                         std::vector<Box>* usedBoxes) const {
  // imgsz 1280 con TTA: es la única configuración que da cajas usables. Bajar a
  // 768 detecta MÁS cajas pero peor puestas, y el OCR consume el recorte —
  // medido, cuesta canción 0.800 -> 0.633.
  //
  // PRIMER PASO: zoom (lejana) o rectify (diagonal). Con --rectify se endereza
  // la pantalla y TODA la lectura corre sobre el warp (la red ve la pantalla
  // axis-aligned, como en entrenamiento); las cajas se mapean de vuelta a la
  // foto solo para el JSON. Si el cuadrilátero no es plausible, cae a zoomIn.
  cv::Mat work = img;
  cv::Mat Hinv;              // warp -> foto; vacío = identidad
  std::vector<Box> boxes;
  // Giro aplicado a `work` para enderezar una foto girada: 0 = original,
  // 1 = 90° horario, 2 = 90° antihorario. Las cajas del JSON y las de
  // `usedBoxes` vuelven a coordenadas de la foto con `turnBack`.
  int turn = 0;
  if (given) {
    boxes = *given;
  } else {
    std::vector<Box> first = det_.detect(img, 1280, augs);
    // Fallback de rotación: si el detector NO vio la pantalla (fullscore), la
    // foto puede estar girada 90° — una foto vertical de una pantalla
    // horizontal, que es lo que ni el detector ni el OCR entienden. Se prueban
    // los dos cuartos de vuelta que faltan, se sigue con la orientación que más
    // ve (fullscore primero) y, si ninguna de las 3 ve nada, no hay lectura.
    // Cuesta dos detecciones extra, y solo en ese caso: con fullscore no corre.
    if (!hasScreen(first)) {
      float bestScore = orientationScore(first);
      for (int t = 1; t <= 2; ++t) {
        cv::Mat rot;
        cv::rotate(img, rot, t == 1 ? cv::ROTATE_90_CLOCKWISE
                                    : cv::ROTATE_90_COUNTERCLOCKWISE);
        std::vector<Box> rb = det_.detect(rot, 1280, augs);
        const float sc = orientationScore(rb);
        if (sc > bestScore) {
          bestScore = sc;
          first = std::move(rb);
          work = rot;
          turn = t;
        }
      }
      if (first.empty()) return emptyJson();
    }
    bool rectified = false;
    if (opts.rectify) {
      const Box* screen = nullptr;
      for (const Box& b : first)
        if (b.cls == 1 && (!screen || b.conf > screen->conf)) screen = &b;
      if (screen) {
        Rectify r = rectifyScreen(work, *screen);
        if (r.ok) { work = r.warp; Hinv = r.Hinv; rectified = true; }
      }
    }
    if (rectified) {
      boxes = det_.detect(work, 1280, augs);
      boxes = zoomTinyTitles(work, boxes);
    } else {
      boxes = zoomIn(work, first);
    }
  }
  auto mb = [&](const Box& b) {
    return turnBack(mapBoxBack(Hinv, b), turn, img.cols, img.rows);
  };
  if (usedBoxes) {
    std::vector<Box> mapped;
    mapped.reserve(boxes.size());
    for (const Box& b : boxes) mapped.push_back(mb(b));
    *usedBoxes = mapped;
  }

  std::ostringstream js;
  js.imbue(std::locale::classic());     // "0.5", nunca "0,5"
  js << "{\"titles\":[";
  std::string chartType;
  float chartConf = 0.f;
  std::vector<int> lvlDigits;
  std::vector<std::vector<float>> lvlScores;

  // song_name: hasta `maxTitleBoxes` cajas, de mayor a menor confianza.
  std::vector<Box> titles;
  for (const Box& b : boxes) if (b.cls == 4) titles.push_back(b);
  std::stable_sort(titles.begin(), titles.end(),
                   [](const Box& a, const Box& b) { return a.conf > b.conf; });
  if (int(titles.size()) > opts.maxTitleBoxes) titles.resize(opts.maxTitleBoxes);
  bool first = true;
  for (const Box& b : titles) {
    cv::Mat roi = cropBox(work, b, 0.04f);
    if (roi.empty()) continue;
    roi = focusBand(roi);
    if (opts.rectify) roi = deskewBand(roi);   // F1.3.5
    auto variants = segmentCharsVariants(roi, 0, opts.binMode);
    if (variants.empty()) continue;
    auto readGlyphs = [&](const std::vector<Glyph>& gs) {
      std::vector<int> lab; std::vector<float> mar;
      chars_.predict(gs, &lab, &mar);
      std::string t;
      for (int L : lab) t += char(L);
      return t;
    };
    const std::string txt = readGlyphs(variants[0]);
    // F4.9: segunda mejor lectura de la misma caja; Kotlin la suma al pool.
    std::string txt2;
    if (opts.maxTitleVariants >= 2 && variants.size() >= 2) {
      txt2 = readGlyphs(variants[1]);
      if (txt2 == txt) txt2.clear();   // no ensuciar el pool con la misma lectura
    }
    if (!first) js << ",";
    js << "{\"raw\":\"" << esc(txt) << "\"";
    if (!txt2.empty()) js << ",\"raw2\":\"" << esc(txt2) << "\"";
    js << ",\"conf\":" << b.conf << ",\"box\":";
    putBox(js, mb(b), img.cols, img.rows);
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
    if (classifyChartType(cropBox(work, *diff, 0.02f), &t, &cf, opts.badgeMode,
                          &chars_)) {
      chartType = t; chartConf = cf;
    }
    auto gs = segmentBadge(cropBox(work, *diff, -0.02f), 2);
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
    readScore(cropBox(work, *scoreBox, 0.06f), digits_, &scoreVal, &scoreMargin,
              &scoreDigits, opts.binMode);

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
  // Grados que se giró `work` para leer (0/90/270). Las cajas ya vienen en
  // coordenadas de la foto original; esto es para que la app pueda mostrar la
  // pantalla derecha en la animación.
  js << ",\"turned_by\":" << (turn == 1 ? 90 : turn == 2 ? 270 : 0);
  js << ",\"screen_box\":";
  if (screenBox) putBox(js, mb(*screenBox), img.cols, img.rows); else js << "null";
  js << ",\"score_box\":";
  if (scoreBox) putBox(js, mb(*scoreBox), img.cols, img.rows); else js << "null";
  js << ",\"badge_box\":";
  if (diff) putBox(js, mb(*diff), img.cols, img.rows); else js << "null";
  js << ",\"rank_box\":";
  if (rankBox) putBox(js, mb(*rankBox), img.cols, img.rows); else js << "null";

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
