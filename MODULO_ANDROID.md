# Módulo Android — COMPILADO

**Estado: el AAR compila y pesa ~22 MB (las cuatro ABIs).** `build/outputs/aar/piu-ocr-release.aar`

Es **un AAR que adentro trae un `.so` compilado con el NDK**. No son alternativas:
el NDK es la herramienta con la que compilás, el AAR es lo que consumís desde
Gradle. La app hace `implementation(project(":piu-ocr"))` y no se entera de que
abajo hay C++.

## 1. Lo que viaja adentro

```
piu-ocr.aar                                        ~22 MB
├── classes.jar                    API Kotlin + song_match
├── jni/{arm64-v8a,armeabi-v7a,x86,x86_64}/
│   └── libpiuocr.so               segment+recognize+text+badge + ncnn + opencv
│                                  (todo linkeado estático en un solo .so por ABI)
└── assets/piu_ocr/
    ├── chars.bin                  2346 plantillas int8         0.41 MB
    ├── digits.bin                 296 ejemplares int8          0.23 MB
    ├── level.bin                  105 plantillas int8          0.01 MB
    ├── catalog.json               675 canciones + charts       0.08 MB
    └── piu_yolo.ncnn.{param,bin}  detector int8                ~3 MB
```

`build_mobile.py` ya genera los tres primeros. Falta solo exportar el YOLO.

**Las cuatro ABIs** (`arm64-v8a`, `armeabi-v7a`, `x86`, `x86_64`). ncnn y
opencv-mobile van estáticos dentro de `libpiuocr.so`, así que cada ABI pesa
~4–14 MB y el AAR se triplica. No es opcional: el bundle de Play exige que todos
los módulos compartan el set de ABIs y la app que consume este AAR ya trae las
cuatro. Si algún día se recorta, hay que bajar el set **también** en la app base
(arm64 + v7 alcanza para dispositivos reales; x86/x86_64 son solo emulador).

## 2. Lo que hay que portar, y lo que no

| archivo | líneas | a dónde va | por qué |
|---|---|---|---|
| `segment.py` | 330 | **C++** | por píxel, es el cuello de latencia |
| `text.py` | 287 | **C++** | idem |
| `recognize.py` | 175 | **C++** | producto punto contra 2346 plantillas |
| `read_score` | — | **C++** (`score.cpp`) | barrido de umbrales guiado por el clasificador |
| `badge.py` | 105 | **C++** | voto de color en HSV |
| `song_match.py` | 107 | **Kotlin** | strings, no es cuello, y evita reescribir `SequenceMatcher` |
| `pipeline.py` | 379 | **Kotlin** | orquestación y gates |

Son ~900 líneas a C++ y **no hay nada de ML ahí**: umbralizar, componentes
conexas, morfología y producto punto.

### OpenCV: solo estas 10 funciones

`cvtColor` · `resize` · `threshold` · `connectedComponentsWithStats` ·
`morphologyEx` · `getStructuringElement` · `GaussianBlur` · `bitwise_not` ·
`warpAffine` · `warpPerspective`

Todas están en **opencv-mobile** (core + imgproc, ~2 MB). No hace falta el SDK
completo de 60 MB. `kmeans` aparece en `recognize.py` pero es solo de build
(`build_mobile.py`), no de inferencia — no va al `.so`.

## 3. Constantes calibradas

Cada una de estas salió de una medición y varias son contraintuitivas. **Si el
port las cambia, el sistema degrada en silencio**: sigue devolviendo respuestas,
solo que peores.

| constante | valor | qué pasa si se toca |
|---|---|---|
| `text.TARGET_H` | 160 | altura de trabajo del título. Sin esto un recorte de foto grande tarda **106 s** |
| `segment.GLYPH_W/H` | 24, 32 | tamaño del glifo normalizado; define el formato de `chars.bin` |
| `segment.BADGE_INNER_R` | 0.66 | máscara del disco de la bolita. A 1.0 el nivel cae de 0.689 a 0.422 |
| `badge.DISC_R` | 0.80 | máscara del voto de color. Más chico no ayuda (0.60 → 0.782) |
| `pipeline.MIN_SONG_MARGIN` | 0.015 | gate de canción. Abajo de esto la precisión se cae a 0.90 |
| `pipeline.MIN_SCORE_MARGIN_SAFE` | 0.010 | gate de score. A 0.02 pierde 15 pts de cobertura |
| `pipeline.MAX_SONG_BOXES` | 3 | fusión de recortes de pantallas 2P |
| `song_match.PREFILTER` | 120 | prefiltro por trigramas antes del matching difuso |
| `text._dominant_line` | h_tol 0.42, gap_mult 2.6 | filtra BPM/FREE PLAY/nombre del jugador. **Vale 15 pts de canción** |
| `text._variants` kernel | `MORPH_ELLIPSE`, k = 0.55×alto | **NO usar `MORPH_RECT`** aunque sea separable: 0.800 → 0.733 |
| percentiles de umbral | 55, 65, 75, 82, 88, 93 | sobre gris y sobre top-hat, 12 combinaciones |
| `segment_badge` | s_max 90, v_min 150, scale 6 | binarización HSV de la bolita |
| pesos del matcher | 0.45 tri + 0.35 ratio + 0.20 partial | mezcla del score de canción |

## 4. YOLO a NCNN

```bash
yolo export model=best.pt format=ncnn imgsz=1280 half=False
```

**Trampa medida:** la cuantización int8 necesita **≥300 imágenes de
calibración**. Un intento previo usó 4 y las confianzas se desplomaron
(`song_name` 0.71 → 0.35), lo que parece "el modelo no sirve en móvil" pero es
calibración. Hoy hay 58 fotos + 200 frames del training set + las que salgan de
los videos.

**Y el detector corre con `augment=True` (TTA) a `imgsz=1280`.** Sin TTA
`song_name` cae de 43 a 29 de 58. NCNN no tiene TTA integrado: hay que hacer las
pasadas a mano (original + escalas + flip) y unir con NMS, o aceptar la pérdida.
Eso es 3× el costo de inferencia del detector, que ya es el 93 % del tiempo.

## 5. La trampa del dynamic feature

Si el módulo va como *dynamic feature* de Play, un `.so` adentro **no carga con
`System.loadLibrary`**. Hay que:

```kotlin
SplitCompat.install(context)
SplitInstallHelper.loadLibrary(context, "piuocr")
```

Falla **solo en builds de Play**, nunca en debug ni en un APK local, así que se
descubre tarde y caro. Si no necesitás la descarga diferida, meterlo como
módulo normal evita el problema entero.

## 6. Presupuesto

| | medido | criterio del plan |
|---|---|---|
| tamaño | 6.5 MB de assets + ~2.5 MB de `.so` | < 15 MB ✔ |
| latencia OCR | 171 ms (Ryzen 7 5700X) | < 100 ms ✖ |

El perfil dice que lo que queda del tiempo es segmentación (`morphologyEx` +
componentes conexas), que es justo lo que el C++ acelera. El matching de
plantillas ya dejó de ser el cuello: comprimirlo de 33 K a 2.3 K bajó la
latencia de 288 a 105 ms solo cuando el top-hat era rectangular.


---

## 7. Cómo se compila

```bash
cd piu_ocr/android
export JAVA_HOME=/home/rodrigo/.jdks/ms-17.0.20.1     # 17 o 21; el 27-ea NO
export ANDROID_HOME=/home/rodrigo/Android/Sdk
gradle assembleRelease
```

Tres cosas que costaron y quedan anotadas:

| síntoma | causa | arreglo |
|---|---|---|
| `FAILURE: 27-ea` | AGP no soporta el JDK 27 | `JAVA_HOME` a un 17 o 21 |
| `undefined symbol: __kmpc_dispatch_deinit` | opencv-mobile linkea `-static-openmp` y el libomp del NDK 27 no lo trae | `ndkVersion = "29.0.14206865"` |
| `Multiple projects have project directory` | `include(":piu-ocr")` apuntando a la raíz | sacarlo del `settings.gradle.kts` |
| `cannot use 'try' with exceptions disabled` | `ncnn.cmake` propaga `-fno-exceptions` por INTERFACE y pisa el `-fexceptions` propio | `set_target_properties(ncnn PROPERTIES INTERFACE_COMPILE_OPTIONS "")` en el CMakeLists |

## 8. Lo que quedó adentro

```
piu-ocr-release.aar                              10.14 MB
├── classes.jar                                   0.03    PiuOcr + SongMatcher
├── jni/arm64-v8a/libpiuocr.so                   12.84    (comprime a ~3)
└── assets/piu_ocr/
    ├── piu_yolo.bin / .param                     6.15    NCNN fp16
    ├── chars.bin                                 1.73    2346 plantillas int8
    ├── level.bin                                 0.11    155 plantillas int8
    └── catalog.json                              0.08    675 canciones
```

Símbolos JNI verificados con `llvm-nm`: `nativeCreate`, `nativeDestroy`, `nativeRead`.

Hoy: `libpiuocr.so` **7.6 MB** (ncnn sin Vulkan, `-fvisibility=hidden`,
`--gc-sections`, `--exclude-libs,ALL`; exporta solo los 3 `Java_*`).

## 9. Test de paridad (host)

`tools/parity/parity.py` compila el mismo C++ como binario Linux y lo corre
sobre las 58 fotos contra el pipeline Python y el GT. Ver `tools/parity/README.md`.
Lo que encontró al primer uso, todo invisible desde Android:

| bug | síntoma | causa |
|---|---|---|
| detector: 0/58 `song_name` | cajas conf 0.9999 clase 1, fuera de la imagen | el export NCNN es de **forma fija** (`Reshape 0=33600`); la pasada TTA a 1056 px devolvía basura. Ahora la escala se hace adentro del lienzo de 1280 |
| `level_digits` = `[50, 54]` | nivel "5054" | `build_mobile.py` guarda las clases como code point: `'2'` → 50. `digitOf()` acepta las dos convenciones |
| chart_type 0.881 → 0.810 | "halfdouble" de más | nativo votaba el mejor entre TODAS las cajas de bolita; una segunda caja floja sobre pantalla azul gana con conf 1.0. Python usa solo la primera |
| Kotlin ≠ Python en 4/90 | canción distinta al borde del gate | `similarity` era LCS y `difflib.ratio()` no lo es; Python redondea el score a 4 decimales; `org.json` de JVM no preserva el orden del catálogo; Kotlin filtraba por chart_type y Python no |
| round() | glifos un píxel distintos | Python redondea el .5 al par; `std::lround` lo aleja de cero (`pyRound`) |

Números de referencia (`baseline.json`), cajas de PyTorch: canción 0.756 de
acierto (igual que Python), nivel 0.810 (Python 0.738), chart 0.837.
End-to-end con el detector NCNN: canción 0.644, recall de `song_name` 0.844.

## 10. Sobre el modelo: fp16, no int8

`ultralytics` **no soporta `int8=True` para formato ncnn** — la cuantización
real necesita `ncnn2table` + `ncnn2int8`, que no vienen en el release de
Android. Se exportó fp16: 11.85 MB → **6.13 MB**, y cuesta 1-3 detecciones
sobre 58 fotos, o sea nada.

**El export a NCNN no pierde precisión.** Medido, detecciones de `song_name`
sobre las 58 fotos:

| | song | difficulty | score | rank |
|---|---|---|---|---|
| PyTorch **con** TTA | **43** | 58 | 56 | 56 |
| PyTorch sin TTA | 29 | 42 | 42 | 34 |
| NCNN fp32 sin TTA | 27 | 43 | 41 | 37 |
| NCNN fp16 sin TTA | 26 | 41 | 39 | 34 |

NCNN sin TTA ≈ PyTorch sin TTA. **Todo lo que se pierde es el TTA**, que
ultralytics aplica en PyTorch y ncnn ignora en silencio — por eso está
implementado a mano en `detector.cpp`.
