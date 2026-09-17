// F1: enderezar la pantalla por perspectiva. El detector devuelve cajas
// axis-aligned y no hay deskew en ninguna parte del pipeline: el título se
// salva con focusBand, pero el score, el nivel y la bolita se leen sobre
// recortes rotados/en perspectiva, donde los glifos salen pegados y la caja de
// la bolita "de lado". Acá se estima el cuadrilátero de la pantalla (brillante
// sobre cabina oscura), se hace warpPerspective a un rectángulo y la red ve la
// pantalla axis-aligned, como en entrenamiento.
//
// Todo opt-in (Options::rectify). Si el cuadrilátero no es plausible se
// devuelve ok=false y el camino actual (zoomIn) sigue intacto. Solo core+imgproc.
#include "piu_ocr.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace piu {
namespace {

float dist(const cv::Point2f& a, const cv::Point2f& b) {
  return std::hypot(a.x - b.x, a.y - b.y);
}

// Ordena 4 puntos como tl, tr, br, bl (arriba por y, desempate por x).
void orderQuad(std::vector<cv::Point2f>* q) {
  std::sort(q->begin(), q->end(),
            [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
  std::vector<cv::Point2f> top{(*q)[0], (*q)[1]}, bot{(*q)[2], (*q)[3]};
  auto byX = [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; };
  std::sort(top.begin(), top.end(), byX);
  std::sort(bot.begin(), bot.end(), byX);
  *q = {top[0], top[1], bot[1], bot[0]};
}

}  // namespace

Box mapBoxBack(const cv::Mat& Hinv, const Box& b) {
  if (Hinv.empty() || Hinv.rows != 3 || Hinv.cols != 3) return b;
  std::vector<cv::Point2f> src = {{float(b.x1), float(b.y1)},
                                  {float(b.x2), float(b.y1)},
                                  {float(b.x2), float(b.y2)},
                                  {float(b.x1), float(b.y2)}};
  std::vector<cv::Point2f> dst;
  cv::perspectiveTransform(src, dst, Hinv);
  float x1 = dst[0].x, x2 = dst[0].x, y1 = dst[0].y, y2 = dst[0].y;
  for (const cv::Point2f& p : dst) {
    x1 = std::min(x1, p.x); x2 = std::max(x2, p.x);
    y1 = std::min(y1, p.y); y2 = std::max(y2, p.y);
  }
  Box o = b;
  o.x1 = int(std::lround(x1)); o.y1 = int(std::lround(y1));
  o.x2 = int(std::lround(x2)); o.y2 = int(std::lround(y2));
  return o;
}

Rectify rectifyScreen(const cv::Mat& bgr, const Box& sb) {
  Rectify r;
  if (bgr.empty()) return r;

  const int bw = sb.x2 - sb.x1, bh = sb.y2 - sb.y1;
  if (bw < 40 || bh < 40) return r;
  const int pad = int(0.06f * std::max(bw, bh));
  const int x1 = std::max(0, sb.x1 - pad), y1 = std::max(0, sb.y1 - pad);
  const int x2 = std::min(bgr.cols, sb.x2 + pad), y2 = std::min(bgr.rows, sb.y2 + pad);
  if (x2 - x1 < 40 || y2 - y1 < 40) return r;

  cv::Mat gray;
  cv::cvtColor(bgr(cv::Rect(x1, y1, x2 - x1, y2 - y1)), gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, gray, {5, 5}, 0);

  // La pantalla es brillante sobre la cabina oscura; Otsu la separa.
  cv::Mat bin;
  cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  cv::morphologyEx(bin, bin, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_RECT, {7, 7}));

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) return r;
  const std::vector<cv::Point>& big = *std::max_element(
      contours.begin(), contours.end(), [](const std::vector<cv::Point>& a,
                                           const std::vector<cv::Point>& b) {
        return cv::contourArea(a) < cv::contourArea(b);
      });
  // Plausibilidad: el contorno debe llenar buena parte del recorte. Si Otsu
  // separó, por ejemplo, la portada del álbum, el área es chica y se aborta.
  const double fill = cv::contourArea(big) / double(std::max<size_t>(1, gray.total()));
  if (fill < 0.30 || fill > 1.02) return r;

  std::vector<cv::Point> hull, approx;
  cv::convexHull(big, hull);
  cv::approxPolyDP(hull, approx, 0.02 * cv::arcLength(hull, true), true);

  std::vector<cv::Point2f> quad;
  if (approx.size() == 4) {
    for (const cv::Point& p : approx) quad.emplace_back(float(p.x + x1), float(p.y + y1));
  } else {
    // Si no son 4 lados, el rectángulo rotado mínimo sigue siendo una pantalla
    // enderezable; es peor que el cuadrilátero pero mejor que no rectificar.
    const cv::RotatedRect rr = cv::minAreaRect(big);
    cv::Point2f pts[4]; rr.points(pts);
    for (int i = 0; i < 4; ++i) quad.emplace_back(pts[i].x + x1, pts[i].y + y1);
  }
  orderQuad(&quad);

  const float wTop = dist(quad[0], quad[1]), wBot = dist(quad[3], quad[2]);
  const float hL = dist(quad[0], quad[3]), hR = dist(quad[1], quad[2]);
  const float W = std::max(wTop, wBot), H = std::max(hL, hR);
  if (W < 60 || H < 60) return r;
  const float asp = W / H;
  // Rango ancho a propósito: la pantalla de Phoenix es apaisada, pero una foto
  // muy en diagonal comprime el lado corto. Fuera de [0.6, 5] sí es basura.
  if (asp < 0.6f || asp > 5.0f) return r;

  const int tw = int(std::min(W, 1600.f)), th = int(std::min(H, 1600.f));
  if (tw < 60 || th < 60) return r;
  const std::vector<cv::Point2f> dst = {{0.f, 0.f}, {float(tw), 0.f},
                                        {float(tw), float(th)}, {0.f, float(th)}};
  const cv::Mat Hm = cv::getPerspectiveTransform(quad, dst);
  cv::Mat hom;
  // Un cuadrilátero casi degenerado da una homografía singular: invertirla
  // podría "funcionar" devolviendo basura y el mapBoxBack de vuelta quedaría
  // mal. Si no se puede invertir, no se rectifica y sigue zoomIn.
  if (cv::invert(Hm, hom) == 0.0) return r;
  cv::Mat warp;
  cv::warpPerspective(bgr, warp, Hm, cv::Size(tw, th), cv::INTER_LINEAR);
  if (warp.empty()) return r;
  r.Hinv = hom;
  r.ok = true;
  r.warp = warp;
  return r;
}

}  // namespace piu
