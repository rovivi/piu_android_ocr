// Puente JNI. Deliberadamente delgado: acá solo se convierte el Bitmap a Mat,
// se corren los campos que viven en C++ y se devuelve JSON. Los gates y el
// matching contra el catálogo viven en Kotlin, porque son lógica de producto
// que conviene poder tocar sin recompilar el .so.
#include "piu_ocr.h"
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <opencv2/imgproc.hpp>
#include <exception>
#include <string>

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "piuocr", __VA_ARGS__)

using namespace piu;

namespace {

// Desbloquea los píxeles también si cvtColor tira: un lockPixels sin su
// unlock deja el Bitmap inutilizable para el resto del proceso.
struct PixelLock {
  JNIEnv* env; jobject bmp; void* px = nullptr;
  PixelLock(JNIEnv* e, jobject b) : env(e), bmp(b) {
    if (AndroidBitmap_lockPixels(env, bmp, &px) < 0) px = nullptr;
  }
  ~PixelLock() { if (px) AndroidBitmap_unlockPixels(env, bmp); }
};

bool bitmapToMat(JNIEnv* env, jobject bmp, cv::Mat* out) {
  AndroidBitmapInfo info;
  if (AndroidBitmap_getInfo(env, bmp, &info) < 0) return false;
  if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    LOGE("bitmap format %d no soportado (se espera ARGB_8888)", info.format);
    return false;
  }
  PixelLock lock(env, bmp);
  if (!lock.px) return false;
  // stride puede ser mayor que width*4: respetarlo.
  cv::Mat rgba(int(info.height), int(info.width), CV_8UC4, lock.px,
               size_t(info.stride));
  cv::cvtColor(rgba, *out, cv::COLOR_RGBA2BGR);
  return true;
}

}  // namespace

extern "C" {

// Devuelve 0 si algún asset no cargó: un handle a medio inicializar solo
// produce lecturas vacías en silencio, que es lo peor que puede pasar.
JNIEXPORT jlong JNICALL
Java_com_piu_ocr_PiuOcr_nativeCreate(JNIEnv* env, jclass, jstring dir) {
  const char* d = env->GetStringUTFChars(dir, nullptr);
  if (!d) return 0;
  const std::string base(d);
  env->ReleaseStringUTFChars(dir, d);

  auto* c = new Engine();
  try {
    if (!c->load(base)) {
      LOGE("nativeCreate: fallo cargando assets en %s", base.c_str());
      delete c;
      return 0;
    }
  } catch (const std::exception& e) {
    LOGE("nativeCreate: %s", e.what());
    delete c;
    return 0;
  }
  return reinterpret_cast<jlong>(c);
}

JNIEXPORT void JNICALL
Java_com_piu_ocr_PiuOcr_nativeDestroy(JNIEnv*, jclass, jlong h) {
  delete reinterpret_cast<Engine*>(h);
}

// Opt-in en runtime (F0.1). Los enteros siguen el orden de los enum de
// piu_ocr.h: BinMode Legacy=0/Clahe=1/Adaptive=2/All=3,
// BadgeMode Color=0/Adaptive=1/Fusion=2. Defaults == comportamiento actual.
JNIEXPORT void JNICALL
Java_com_piu_ocr_PiuOcr_nativeSetOptions(JNIEnv*, jclass, jlong h, jboolean rectify,
                                         jint binMode, jint badgeMode,
                                         jint titleVariants, jint titleBoxes) {
  auto* c = reinterpret_cast<Engine*>(h);
  if (!c) return;
  c->opts.rectify = rectify;
  if (binMode >= 0 && binMode <= int(BinMode::All))
    c->opts.binMode = static_cast<BinMode>(binMode);
  if (badgeMode >= 0 && badgeMode <= int(BadgeMode::Fusion))
    c->opts.badgeMode = static_cast<BadgeMode>(badgeMode);
  c->opts.maxTitleVariants = titleVariants < 1 ? 1 : titleVariants;
  c->opts.maxTitleBoxes = titleBoxes < 1 ? 3 : titleBoxes;
}

/**
 * Corre el detector y el OCR sobre el bitmap y devuelve JSON. Los gates y el
 * matching contra el catálogo viven en Kotlin: son lógica de producto que
 * conviene poder tocar sin recompilar el .so.
 */
JNIEXPORT jstring JNICALL
Java_com_piu_ocr_PiuOcr_nativeRead(JNIEnv* env, jclass, jlong h, jobject bmp) {
  auto* c = reinterpret_cast<Engine*>(h);
  cv::Mat img;
  if (!c || !bitmapToMat(env, bmp, &img)) return env->NewStringUTF(Engine::emptyJson());
  // Una cv::Exception que cruza la frontera JNI aborta el proceso entero.
  try {
    return env->NewStringUTF(c->read(img).c_str());
  } catch (const std::exception& e) {
    LOGE("nativeRead: %s", e.what());
    return env->NewStringUTF(Engine::emptyJson());
  }
}

}  // extern "C"
