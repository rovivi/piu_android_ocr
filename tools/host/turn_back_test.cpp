// Chequeo del mapeo de cajas del fallback de rotación de Engine::read
// (pipeline.cpp). `turnBack` es inline y `Box` es header-only, así que no hace
// falta linkear OpenCV: alcanza con los headers de opencv-mobile.
//
//   c++ -std=c++17 -I src/main/cpp \
//       -I third_party/opencv-mobile-4.13.0-android/sdk/native/jni/include \
//       tools/host/turn_back_test.cpp -o /tmp/turn_back_test && /tmp/turn_back_test
#include "piu_ocr.h"
#include <cassert>
#include <cstdio>

using namespace piu;

// El giro directo que hace cv::rotate (original -> girada), para poder exigir
// que turnBack sea su inversa.
static Box turnFwd(const Box& b, int turn, int W, int H) {
  if (turn == 1) return Box{H - b.y2, b.x1, H - b.y1, b.x2, b.conf, b.cls};
  if (turn == 2) return Box{b.y1, W - b.x2, b.y2, W - b.x1, b.conf, b.cls};
  return b;
}

int main() {
  const int W = 6, H = 4;
  const Box boxes[] = {{0, 0, W, H, 0.9f, 1}, {1, 1, 4, 3, 0.5f, 4}, {2, 0, 3, 4, 0.7f, 3}};
  for (const Box& b : boxes)
    for (int t : {0, 1, 2}) {
      const Box back = turnBack(turnFwd(b, t, W, H), t, W, H);
      assert(back.x1 == b.x1 && back.y1 == b.y1 && back.x2 == b.x2 && back.y2 == b.y2);
      assert(back.cls == b.cls && back.conf == b.conf);
    }
  // Punto concreto: en 90° horario el píxel (0,0) de la foto cae en la esquina
  // superior derecha de la girada (H-1, 0) y su caja vuelve a (0,0,1,1).
  const Box p{0, 0, 1, 1, 1.f, 0};
  const Box g = turnFwd(p, 1, W, H);
  assert(g.x1 == H - 1 && g.y1 == 0);
  const Box r = turnBack(g, 1, W, H);
  assert(r.x1 == 0 && r.y1 == 0 && r.x2 == 1 && r.y2 == 1);
  std::puts("turn_back: ok");
  return 0;
}
