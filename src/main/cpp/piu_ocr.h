// Puerto a C++ de la cadena de inferencia. Nada de ML: umbralizar, componentes
// conexas, morfología y producto punto. Solo core+imgproc de opencv-mobile.
//
// Las constantes son el resultado de mediciones, varias contraintuitivas.
// Cambiarlas degrada el sistema EN SILENCIO: sigue respondiendo, solo que peor.
// Ver android/MODULO_ANDROID.md §3 antes de tocar cualquiera.
#pragma once
#include <opencv2/core.hpp>
#include <cmath>

namespace ncnn { class Net; }
#include <string>
#include <vector>

namespace piu {

constexpr int   GLYPH_W = 24, GLYPH_H = 32;
constexpr int   TARGET_H = 160;      // altura de trabajo del título
constexpr float MIN_SCALE = 0.15f;
constexpr float BADGE_INNER_R = 0.66f;   // máscara del disco para los dígitos
constexpr float DISC_R = 0.80f;          // máscara del disco para el color
constexpr int   PCTS[] = {55, 65, 75, 82, 88, 93};
constexpr float DOM_H_TOL = 0.42f, DOM_GAP_MULT = 2.6f;
constexpr int   BADGE_S_MAX = 90, BADGE_V_MIN = 150, BADGE_SCALE = 6;

struct Box { int x1, y1, x2, y2; float conf; int cls = -1; };

// round() de Python redondea el .5 al PAR (banker's); std::lround lo aleja
// de cero. Donde el port replica un round() de Python va esto, o los glifos
// salen un píxel distintos y el test de paridad lo marca.
inline int pyRound(double v) {
  const double f = std::floor(v), d = v - f;
  if (d > 0.5) return int(f) + 1;
  if (d < 0.5) return int(f);
  return (int(f) % 2 == 0) ? int(f) : int(f) + 1;
}
// Mediana estilo numpy: promedio de los dos del medio si n es par.
float median(std::vector<float> v);
// Máscara elíptica inscrita en HxW con radio relativo r, con la misma fórmula
// que el numpy de badge.py/segment.py (cv::ellipse rasteriza distinto).
cv::Mat discMask(int H, int W, float r);
struct Glyph { unsigned char px[GLYPH_H * GLYPH_W]; };

// --- detector.cpp ------------------------------------------------------
// clases: 0 difficulty, 1 fullscore, 2 rank, 3 score, 4 song_name
// Una pasada del TTA: escala relativa dentro del lienzo y flip horizontal.
struct Aug { float scale; bool flip; };
// Las 3 pasadas por defecto (original, 0.83, flip). Ver detector.cpp.
const std::vector<Aug>& defaultAugs();

class Detector {
 public:
  ~Detector();
  bool load(const std::string& param, const std::string& bin);
  // Corre una pasada por Aug y las une con NMS. Sin TTA se pierde un tercio
  // de los song_name (43 -> 29 de 58, medido); ncnn no lo trae, va a mano.
  std::vector<Box> detect(const cv::Mat& bgr, int imgsz = 1280,
                          const std::vector<Aug>& augs = defaultAugs()) const;
 private:
  std::vector<Box> detectOnce(const cv::Mat&, int, float, bool) const;
  ncnn::Net* net_ = nullptr;
};

// --- segment.cpp -------------------------------------------------------
cv::Mat cropBox(const cv::Mat& img, const Box& b, float pad = 0.06f,
                float padX = -1.f);
// Dígitos de la bolita. wantN>0 fuerza esa cantidad (el catálogo sabe cuántos).
std::vector<Glyph> segmentBadge(const cv::Mat& roi, int wantN = 0);

// --- text.cpp ----------------------------------------------------------
// Reencuentra la banda del título dentro de un recorte flojo.
cv::Mat focusBand(const cv::Mat& roi);
// Corrida de glifos de altura y separación coherentes: descarta BPM,
// "FREE PLAY" y el nombre del jugador. Vale 15 puntos de acierto.
std::vector<cv::Rect> dominantLine(std::vector<cv::Rect> boxes);
std::vector<Glyph> segmentChars(const cv::Mat& roi, int wantN = 0);

// --- badge.cpp ---------------------------------------------------------
// naranja/rojo=single, verde=double, azul=halfdouble, amarillo=coop
bool classifyChartType(const cv::Mat& roi, std::string* out, float* conf);

// --- recognize.cpp -----------------------------------------------------
// Etiqueta de plantilla -> dígito 0..9, o -1. level.bin trae code points.
int digitOf(int label);
class Templates {
 public:
  static Templates load(const std::string& path);   // chars.bin / level.bin
  // Similitud coseno por clase. margin = mejor - segunda.
  void predict(const std::vector<Glyph>&, std::vector<int>* labels,
               std::vector<float>* margins) const;
  // Similitud por clase sin colapsar: la necesita el cruce con el catálogo,
  // que reordena niveles enteros y no solo acepta o rechaza el argmax.
  void scores(const std::vector<Glyph>&, std::vector<std::vector<float>>*) const;
  bool empty() const { return t_.empty(); }
 private:
  cv::Mat perClass(const std::vector<Glyph>&) const;
  cv::Mat t_;                 // (K, GLYPH_H*GLYPH_W) float32 L2-normalizadas
  std::vector<int> classes_;  // etiqueta por plantilla
  std::vector<int> uniq_;     // etiquetas distintas, ordenadas
  std::vector<int> clsIdx_;   // índice en uniq_ por plantilla (precalculado)
};

// --- score.cpp ---------------------------------------------------------
// Dígitos de una franja de una sola línea. Necesita el clasificador para
// elegir el umbral: la regularidad sola no distingue un dígito de un borrón.
std::vector<Glyph> segmentDigits(const cv::Mat& roi, const Templates& digits,
                                 int scale = 4);
// Score de PIU Phoenix, 0..1000000. false si no se pudo leer.
bool readScore(const cv::Mat& roi, const Templates& digits, int* value,
               float* margin, std::string* rawDigits);

// --- pipeline.cpp ------------------------------------------------------
// Lo que corre nativeRead, sin JNI: lo comparten el .so y el CLI de host
// (tools/host) que alimenta el test de paridad contra el pipeline Python.
class Engine {
 public:
  bool load(const std::string& assetDir);
  // Detecta con YOLO y lee. `boxes` no nulo saltea el detector y usa esas
  // cajas (el test de paridad las toma de dataset_v2/boxes.json).
  std::string read(const cv::Mat& bgr, const std::vector<Box>* boxes = nullptr,
                   std::vector<Box>* usedBoxes = nullptr) const;
  static const char* emptyJson();
  std::vector<Aug> augs = defaultAugs();     // el CLI las cambia para medir
 private:
  Templates chars_, level_, digits_;
  Detector det_;
  // Segunda detección dentro de la pantalla (fullscore) cuando ocupa poco del
  // encuadre: una foto lejana. Devuelve cajas en coordenadas de la imagen.
  std::vector<Box> zoomIn(const cv::Mat& img, std::vector<Box> first) const;
};

}  // namespace piu
