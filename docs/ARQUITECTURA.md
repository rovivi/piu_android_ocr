# Arquitectura

Cómo está hecho el módulo por dentro y por qué. Para el uso desde una app ver
[API.md](API.md); para correr los tests ver [VERIFICACION.md](VERIFICACION.md);
para los resultados medidos ver [INFORME.md](INFORME.md).

> **Si vas a tocar una constante del C++ o un gate de Kotlin, leé la sección
> [7](#7-constantes-calibradas) y [8](#8-gates-kotlin) primero.** Casi todas
> salieron de una medición y varias son contraintuitivas: cambiarlas degrada el
> sistema **en silencio** (sigue respondiendo, solo que peor).

## Índice

1. [Visión general](#1-visión-general)
2. [Qué corre en C++ y qué en Kotlin](#2-qué-corre-en-c-y-qué-en-kotlin)
3. [El AAR por dentro](#3-el-aar-por-dentro)
4. [Dependencias nativas](#4-dependencias-nativas)
5. [Detector YOLO y TTA](#5-detector-yolo-y-tta)
6. [OCR clásico](#6-ocr-clásico)
7. [Constantes calibradas](#7-constantes-calibradas)
8. [Gates Kotlin](#8-gates-kotlin)
9. [Modos opt-in](#9-modos-opt-in)
10. [Compilación](#10-compilación)
11. [Problemas conocidos](#11-problemas-conocidos)
12. [Presupuesto y latencia](#12-presupuesto-y-latencia)

---

## 1. Visión general

Este repo es **el módulo Android**: un AAR que adentro trae un `.so` compilado
con el NDK. No son alternativas — el NDK es la herramienta con la que se
compila, el AAR es lo que consume Gradle. La app hace
`implementation(project(":piu-ocr"))` y no se entera de que abajo hay C++.

El flujo completo, de la foto al dato:

```
Bitmap ──▶ nativeRead (JNI)
             │
             ├─ (opt) rectify      endereza la pantalla por perspectiva
             ├─ Detector YOLO      NCNN fp16 + 3 pasadas TTA + NMS
             │                     (o zoomIn si la pantalla está lejos)
             │
             ├─ title  → focusBand → segmentChars → plantillas chars.bin
             ├─ badge  → color HSV → sub-label (modo Fusion)
             ├─ level  → segmentBadge → plantillas level.bin
             └─ score  → segmentDigits → plantillas digits.bin
             │
             ▼  JSON crudo
Kotlin interpret()
             ├─ gate de chart  (MIN_BADGE_CONF)
             ├─ matcher difuso contra catalog.json (SongMatcher)
             ├─ gate de canción (margen + looksLike)
             ├─ reordena niveles legales según el catálogo
             └─ gate de score  (MIN_SCORE_MARGIN)
             │
             ▼
Reading  →  la app guarda o escala a un VLM
```

El OCR **no tiene nada de ML**: umbralizar, componentes conexas, morfología y
producto punto. El único modelo es el detector.

## 2. Qué corre en C++ y qué en Kotlin

La línea divisoria es deliberada: **C++ para lo que es por píxel** (el cuello de
latencia) y **Kotlin para lo que es lógica de producto** (se toca sin recompilar
el `.so`).

| pieza | líneas | dónde | por qué |
|---|---:|---|---|
| `detector.cpp` | ~160 | **C++** | YOLO en NCNN + TTA a mano |
| `segment.cpp` | ~165 | **C++** | por píxel, cuello de latencia |
| `text.cpp` | ~315 | **C++** | idem |
| `recognize.cpp` | ~150 | **C++** | producto punto contra ~2.3 K plantillas |
| `score.cpp` | ~305 | **C++** | barrido de umbrales guiado por el clasificador |
| `badge.cpp` | ~125 | **C++** | voto de color en HSV |
| `rectify.cpp` | ~130 | **C++** | enderezado por perspectiva (opt-in) |
| `SongMatcher.kt` | ~250 | **Kotlin** | strings; evita reescribir `SequenceMatcher` |
| `PiuOcr.kt` | ~370 | **Kotlin** | orquestación, gates y cruce con el catálogo |

El `jni_bridge.cpp` es deliberadamente delgado: convierte el Bitmap a `cv::Mat`,
corre `Engine::read` y devuelve JSON.

## 3. El AAR por dentro

Medido sobre `build/outputs/aar/piu-ocr-release.aar` (**~22 MB**, cuatro ABIs):

```
piu-ocr-release.aar                          21.9 MB
├── classes.jar                              0.04 MB   API Kotlin + SongMatcher
├── jni/
│   ├── arm64-v8a/libpiuocr.so               7.6 MB
│   ├── armeabi-v7a/libpiuocr.so             4.0 MB
│   ├── x86/libpiuocr.so                    12.9 MB
│   └── x86_64/libpiuocr.so                 14.1 MB
└── assets/piu_ocr/
    ├── piu_yolo.bin                         6.43 MB   detector NCNN fp16
    ├── chars.bin                            1.81 MB   ~2.3 K plantillas int8
    ├── digits.bin                           0.23 MB   296 ejemplares de score
    ├── level.bin                            0.12 MB   plantillas de nivel
    └── catalog.json                         0.08 MB   675 canciones + charts
```

Cada `.so` lleva ncnn y opencv-mobile **linkeados estáticos adentro**, así que
cada ABI pesa lo suyo y el AAR se multiplica.

**Por qué las cuatro ABIs y no solo arm64:** el bundle de Play exige que *todos*
los módulos compartan el mismo set de ABIs, y la app que consume este AAR ya
trae las cuatro de sus propias dependencias. Con solo arm64 acá, `bundleRelease`
falla con *"All modules with native libraries must support the same set of
ABIs"*. Si algún día se recorta, hay que bajar el set **también** en la app base
(arm64 + v7 alcanza para dispositivos reales; x86/x86_64 son solo emulador).

## 4. Dependencias nativas

### OpenCV: solo `core` + `imgproc`

El SDK completo son ~60 MB; acá se usa un subconjunto que entra en
**opencv-mobile** (~2 MB). Funciones usadas, por categoría:

| categoría | funciones |
|---|---|
| color / tamaño | `cvtColor` · `resize` · `split` · `GaussianBlur` |
| umbral | `threshold` · `adaptiveThreshold` · `createCLAHE` · `bitwise_not` |
| componentes | `connectedComponentsWithStats` · `morphologyEx` · `getStructuringElement` |
| geometría | `warpAffine` · `warpPerspective` · `getRotationMatrix2D` · `getPerspectiveTransform` · `perspectiveTransform` · `invert` |
| contornos | `findContours` · `convexHull` · `approxPolyDP` · `contourArea` · `arcLength` · `minAreaRect` |
| bordes | `Canny` · `HoughLinesP` |
| varios | `reduce` · `minMaxLoc` · `countNonZero` · `gemm` |

`kmeans` aparece en el pipeline Python pero es **solo de build** (`build_mobile.py`),
no de inferencia: no va al `.so`. `highgui`/`imread` los usa únicamente el CLI de
host, nunca el módulo Android.

### ncnn sin Vulkan

El detector corre con `use_vulkan_compute = false`, así que glslang + SPIRV +
MachineIndependent (~5.3 MB de `.a`) eran peso muerto y se excluyen.

## 5. Detector YOLO y TTA

Clases: `0 difficulty`, `1 fullscore`, `2 rank`, `3 score`, `4 song_name`.

### El TTA va a mano

El export a NCNN es de **forma fija** (los `Reshape` del `.param` llevan
`0=33600` anchors, los de 1280). Alimentar 1056 px devolvía basura con conf
0.9999 en la clase 1 que después ganaba el NMS: 0 de 58 `song_name`. Por eso la
reducción se hace **adentro del lienzo de 1280** (la imagen ocupa el 83 % y el
resto es relleno 114), que es la misma augmentación sin cambiar la forma de
entrada.

Sin TTA, un tercio de los títulos no se detecta. Medido sobre 58 fotos, 45 con
ground truth (`tools/parity --detect --augs`), recall de `song_name` y acierto
de canción end-to-end:

| pasadas | recall song_name | acierto canción | ms (host) |
|---|---|---:|---:|
| `1` | 0.733 | 0.444 | 276 |
| `1, 1f` | 0.756 | 0.489 | 481 |
| `1, 0.83, 1f` | 0.844 | 0.644 | 661 |
| `1, 0.83f` | 0.933 | 0.733 | 486 |
| **`1, 0.83, 0.83f`** | **0.956** | **0.733** | 653 |
| `1, 0.83, 1f, 0.83f` | 0.933 | 0.733 | 844 |

El flip sin reducir casi no aporta; el flip **de la reducida** sí. Las tres
pasadas por defecto cuestan 3× detector, que ya es el ~90 % del tiempo: es la
decisión de latencia más cara del módulo y está tomada a conciencia.

### Zoom para fotos lejanas

Si el detector marcó la pantalla entera (`fullscore`, recall 1.0) y ocupa menos
del 40 % del encuadre, se vuelve a detectar **solo dentro de ella**
(`Engine::zoomIn`). Para la red es la misma pantalla llenando el lienzo, y las
cajas vuelven en coordenadas de la foto.

### NMS

Se corre por clase sobre la unión de las pasadas (IoU > 0.45). Con umbral de
confianza 0.005 las tres pasadas juntan cientos de cajas, así que se ordena por
`(clase, conf)` para que cada clase quede contigua.

### Por qué fp16 y no int8

`ultralytics` **no soporta `int8=True` para formato ncnn**: la cuantización real
necesita `ncnn2table` + `ncnn2int8`, que no vienen en el release de Android. Se
exportó fp16: 11.85 MB → **6.13 MB**, y cuesta 1–3 detecciones sobre 58 fotos.

El export a NCNN **no pierde precisión**; todo lo que se pierde es el TTA:

| | song | difficulty | score | rank |
|---|---:|---:|---:|---:|
| PyTorch **con** TTA | **43** | 58 | 56 | 56 |
| PyTorch sin TTA | 29 | 42 | 42 | 34 |
| NCNN fp32 sin TTA | 27 | 43 | 41 | 37 |
| NCNN fp16 sin TTA | 26 | 41 | 39 | 34 |

```bash
yolo export model=best.pt format=ncnn imgsz=1280 half=False
```

> **Trampa medida:** la cuantización int8 necesita **≥300 imágenes de
> calibración**. Un intento previo usó 4 y las confianzas se desplomaron
> (`song_name` 0.71 → 0.35), lo que parece "el modelo no sirve en móvil" pero es
> calibración.

## 6. OCR clásico

### Título (`text.cpp`)

1. **`focusBand`**: reencuentra la banda del título dentro del recorte. Las
   filas de afuera son portada del álbum, que envenena los umbrales.
2. **`dominantLine`**: se queda con la corrida de glifos de altura y separación
   coherentes; descarta BPM, `FREE PLAY` y el nombre del jugador. **Vale 15
   puntos de acierto** y además hace que el lector no dependa de que la caja de
   YOLO esté bien ajustada a lo ancho.
3. **`segmentChars`**: umbraliza por percentil y top-hat, extrae componentes y
   normaliza cada glifo a `24×32`. En `BinMode::All` agrega CLAHE y
   `adaptiveThreshold`; el mismo puntaje de coherencia elige.
4. **`segmentCharsVariants`** devuelve las dos mejores lecturas de la misma caja
   (alimenta el `raw2` de F4).

El top-hat usa `MORPH_ELLIPSE`: `MORPH_RECT` sería separable y ~14 ms más
barato, pero es medible peor (0.733 vs 0.800) porque las esquinas del rectángulo
llegan más lejos en diagonal y comen trazo.

La altura de trabajo se normaliza a `TARGET_H = 160`. Sin eso, un recorte de una
foto de 2048 px daba una imagen de 4500×900 umbralizada 12 veces: **106 segundos**.

### Bolita / chart type (`badge.cpp`, `segment.cpp`)

- Se enmascara el **disco inscrito** (`DISC_R = 0.80`): las esquinas de la caja
  son pantalla y en el layout Phoenix son azules, con lo que 39 de 84 bolitas
  salían `halfdouble`. Restringir el voto al disco lo arregla (0.500 → 0.833).
- El voto de color es por conteo de tono en rango (naranja/rojo = single,
  verde = double, azul/cian = halfdouble, amarillo = coop).
- `segmentBadge` enmascara un disco interior (`BADGE_INNER_R = 0.66`) para que
  el borde brillante no salga como falso dígito, y `forceN` fuerza la cantidad
  de dígitos que el catálogo sabe que hay.

### Score (`score.cpp`)

- Barrido de umbrales (Otsu, su inverso y percentiles), filtrado de componentes
  por altura, relleno, proporción y **brillo** (PIU dibuja los ceros no
  significativos en gris apagado), y la línea del score separada del bonus
  `+84,818` que va debajo.
- El umbral se elige **preguntándole al clasificador**: la regularidad sola no
  distingue un `8` de un borrón relleno por un umbral malo — los dos son un blob
  del tamaño correcto.
- Regla de dominio: `0..1 000 000`; el único valor legítimo de 7 dígitos es
  `1000000`, cualquier otro significa que se coló un dígito de más.

### Enderezado (`rectify.cpp`, opt-in)

Estima el cuadrilátero de la pantalla (brillante sobre cabina oscura → Otsu →
contorno más grande → `approxPolyDP`/`minAreaRect`), valida plausibilidad (área,
aspecto, homografía invertible) y hace `warpPerspective`. La red pasa a ver la
pantalla axis-aligned, como en entrenamiento, y `mapBoxBack` mapea las cajas de
vuelta a la foto para el JSON. Si algo no cierra, devuelve `ok=false` y sigue el
camino `zoomIn` intacto.

## 7. Constantes calibradas

Cada una salió de una medición; varias son contraintuitivas. Las del C++ viven
en `src/main/cpp/piu_ocr.h` y en el archivo que las usa.

| constante | valor | qué pasa si se toca |
|---|---|---|
| `text.TARGET_H` | 160 | altura de trabajo del título. Sin esto un recorte grande tarda **106 s** |
| `segment.GLYPH_W/H` | 24, 32 | tamaño del glifo normalizado; define el formato de `chars.bin` |
| `segment.BADGE_INNER_R` | 0.66 | máscara del disco de la bolita. A 1.0 el nivel cae de 0.689 a 0.422 |
| `badge.DISC_R` | 0.80 | máscara del voto de color. Más chico no ayuda (0.60 → 0.782) |
| `detector.kAugs` | `{1, 0.83, 0.83f}` | pasadas del TTA. Ver §5 |
| `pipeline.ZOOM_MAX_SHARE` | 0.40 | debajo de esta fracción del encuadre la pantalla está "lejos" |
| `text._dominant_line` | h_tol 0.42, gap_mult 2.6 | filtra BPM/FREE PLAY/nombre del jugador. **Vale 15 pts de canción** |
| `text._variants` kernel | `MORPH_ELLIPSE`, k = 0.55×alto | **NO usar `MORPH_RECT`** aunque sea separable: 0.800 → 0.733 |
| percentiles de umbral | 55, 65, 75, 82, 88, 93 | sobre gris y sobre top-hat |
| `segment_badge` | s_max 90, v_min 150, scale 6 | binarización HSV de la bolita |
| `score` | Otsu + inv + {60, 70, 78, 85, 90, 94} | barrido de umbrales de la franja |
| pesos del matcher | 0.45 tri + 0.35 ratio + 0.20 partial | mezcla del score de canción |

## 8. Gates Kotlin

Viven en `PiuOcr.kt` y en `SongMatcher.kt`. La réplica Python del test de
paridad (`tools/parity/parity.py`) los repite, y `ParityTest.kt` exige que
coincidan.

| gate | valor | medición |
|---|---:|---|
| `MIN_SONG_MARGIN` | 0.010 | 0.010 → cob 0.889 / prec 0.900; 0.015 → cob 0.800 / prec 0.972 |
| `MIN_SCORE_MARGIN` | 0.010 | 0.010 → cob 0.94 / prec 0.98; 0.000 → cob 1.00 / prec 0.94 |
| `MIN_BADGE_CONF` | 0.15 | 0.15 → 0.873; 0.35 → 0.655 (convierte lecturas buenas en null) |
| `MIN_RAW_SIMILARITY` | 0.55 | 0.50 → 66 bien/4 mal; 0.55 → 63 bien/2 mal; 0.70 → 50 bien/0 mal |
| `MIN_RAW_SIMILARITY_SHORT` | 0.80 | un nombre de ≤3 letras ("Bee", "N") exige casi coincidencia |
| `SongMatcher.PREFILTER` | 120 | prefiltro por trigramas antes del matching difuso |
| `MAX_SONG_BOXES` | 5 | fusión de recortes de pantallas 2P (default C++: 3) |

Dos gates de canción, no uno: el **margen** dice que el pool está decidido; el
**parecido** (`looksLike`) dice que lo que ganó se parece a lo leído. Sin el
segundo, el pool ordena bien y separa mal — los ocho errores del fixture eran
nombres cortos que ganan sobre glifos basura porque todo se parece un poco a
`N`.

`SongMatcher` usa `SequenceMatcher.ratio()` de difflib porteado tal cual (no es
LCS) y redondea a 4 decimales, porque el gate de margen se decide en el cuarto
decimal.

## 9. Modos opt-in

Todo mecanismo nuevo entra **opt-in con flags** y el **default no cambia** hasta
que la suite de paridad confirme un ganador. Eso evita degradar en silencio el
baseline. Ver [PLAN_LUZ_ANGULO.md](PLAN_LUZ_ANGULO.md).

| flag | `Options` | default | qué hace |
|---|---|---|---|
| `--rectify` | `rectify` | `false` | endereza la pantalla por perspectiva (F1) |
| `--bin-mode` | `binMode` | `Legacy` | `Legacy`/`Clahe`/`Adaptive`/`All` (F2) |
| `--badge-mode` | `badgeMode` | `Color` | `Color`/`Adaptive`/`Fusion` (F3) |
| `--title-variants` | `maxTitleVariants` | 1 | 2 emite `raw2` por caja (F4) |
| `--title-boxes` | `maxTitleBoxes` | 3 | máximo de cajas de título (F4) |

Desde la app se activan con `PiuOcr.setOptions(...)`; desde el host, con las
mismas banderas en `piuocr_cli`.

## 10. Compilación

```bash
export JAVA_HOME=~/.jdks/ms-17.0.20.1       # JDK 17 o 21; el 27-ea NO
export ANDROID_HOME=~/Android/Sdk
./gradlew assembleRelease                    # build/outputs/aar/piu-ocr-release.aar
```

| síntoma | causa | arreglo |
|---|---|---|
| `FAILURE: 27-ea` | AGP no soporta el JDK 27 | `JAVA_HOME` a un 17 o 21 |
| `undefined symbol: __kmpc_dispatch_deinit` | opencv-mobile linkea `-static-openmp` y el libomp del NDK 27 no lo trae | `ndkVersion = "29.0.14206865"` |
| `Multiple projects have project directory` | `include(":piu-ocr")` apuntando a la raíz | sacarlo del `settings.gradle.kts` |
| `cannot use 'try' with exceptions disabled` | `ncnn.cmake` propaga `-fno-exceptions` por INTERFACE y pisa el `-fexceptions` propio | `set_target_properties(ncnn PROPERTIES INTERFACE_COMPILE_OPTIONS "")` |

El `.so` se compila con `-O3 -ffast-math -fexceptions -fvisibility=hidden
--gc-sections --exclude-libs,ALL`: exporta solo los tres `Java_*` y así
`libpiuocr.so` arm64 queda en **7.6 MB** (contra 12.8 MB con símbolos de debug).

## 11. Problemas conocidos

### Dynamic feature de Play

Si el módulo va como *dynamic feature*, un `.so` adentro **no carga con
`System.loadLibrary`**:

```kotlin
SplitCompat.install(context)
SplitInstallHelper.loadLibrary(context, "piuocr")
```

Falla **solo en builds firmados de Play**, nunca en debug ni en un APK local. El
puente lo resuelve por reflexión (ver `ensureLoaded` en `PiuOcr.kt`), así que el
módulo sigue funcionando embebido normal y no obliga a depender de Play Core.

### Otras

- **Fotos HARDWARE**: `ImageDecoder` devuelve bitmaps `HARDWARE` (API 28+) que
  fallan en `lockPixels` en silencio. `PiuOcr.read` los copia a `ARGB_8888`.
- **Actualización del APK**: los assets se copian una vez **por versión** a
  `filesDir` y cada archivo se escribe a `.tmp` y se renombra, para que un crash
  a mitad de copia no deje un `.bin` truncado que "existe".
- **`levelsFor`**: un nivel mal leído no debe borrar la canción correcta. El
  cruce canción↔nivel vale +3 pts cuando el nivel es correcto y es pérdida neta
  cuando no; por eso `match` no filtra por nivel.

## 12. Presupuesto y latencia

| | medido | criterio del plan |
|---|---|---|
| tamaño de assets | ~8.7 MB (con el detector) | < 15 MB ✔ |
| latencia OCR | 674 ms media, 785 ms máx (host, 3 pasadas) | < 100 ms ✖ |
| `libpiuocr.so` arm64 | 7.6 MB | — |

El detector es el ~90 % del tiempo y el TTA es 3× de eso. Las palancas reales,
en orden: cuantización int8 **real** (`ncnn2int8`, ≥300 imágenes de calibración),
menos pasadas del TTA (2 en vez de 3 cuesta 26 % menos con algo menos de
precisión), y recortar el set de ABIs.
