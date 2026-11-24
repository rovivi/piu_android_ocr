# Plan de mejora: detección y OCR bajo luz y ángulo

**Alcance acordado:** solo runtime Android (este repo). No se retrena el YOLO;
el dataset_v2, el venv de `piu_yolo` y la validación end-to-end quedan en la
máquina que tiene los datos. Latencia: sin límite (maximizar aciertos).
Prioridad: canción, chart/nivel y score, todas por igual.

Todo lo nuevo entra **opt-in con flags** y el **default no cambia** hasta que la
suite de paridad (corriendo en la máquina con datos) confirme un ganador.
Eso evita degradar en silencio los números de
[tools/parity/baseline.json](../tools/parity/baseline.json)
(song 0.733, level 0.714, chart_type 0.767, score 0.905).

---

## 0. Estado

El código de todas las fases **ya está implementado y es opt-in**; lo que falta
es medir el A/B sobre el dataset y flipear el default del ganador.

| fase | estado | dónde |
|---|---|---|
| F0.1 flags en `Options` + CLI + `setOptions` | ✅ hecho | `piu_ocr.h`, `cli.cpp`, `PiuOcr.kt` |
| F0.2 buckets por condición | ✅ hecho | `parity.py --buckets` |
| F0.3 smoke sintético | ✅ hecho | `tools/synth.py`, `parity.py --synth` |
| F1 rectify por perspectiva | ✅ implementado, opt-in | `rectify.cpp`, `--rectify` |
| F1.3.5 deskew fino del título | ✅ hecho | `text.cpp::deskewBand` |
| F2 binarización CLAHE/adaptive | ✅ implementada, opt-in | `text.cpp`, `score.cpp`, `--bin-mode` |
| F3.7 hue adaptativo | ✅ implementado, opt-in | `badge.cpp`, `--badge-mode adaptive` |
| F3.8 fusión color ⊕ texto | ✅ implementada, opt-in | `badge.cpp`, `--badge-mode fusion` |
| F4.9 2ª lectura de título (`raw2`) | ✅ implementada, opt-in | `pipeline.cpp`, `--title-variants 2` |
| F4.10 altura mínima de título | ✅ implementada (solo camino rectify) | `zoomTinyTitles` |
| F5 CLAHE/adaptive en el score | ✅ implementada, opt-in | `score.cpp` |
| medir A/B y flipear defaults | ⬜ pendiente (máquina con datos) | §4 |

Los pasos concretos para cerrar el plan están en [§4](#4-validación) y
[§6](#6-orden-de-ejecución). El diseño interno de los flags está en
[ARQUITECTURA.md §9](ARQUITECTURA.md#9-modos-opt-in).

---

## 1. Diagnóstico con evidencia

### 1.1 Luz

- Las 178 imágenes de entrenamiento (`piu_yolo/images`) son **todas oscuras**:
  brillo medio ~73/255, 43 fotos < 60, **ninguna > 125**. El detector casi no
  vio escenas brillantes/lavadas ni glare de neón; y la máquina real falla más
  en luz irregular que en oscuridad pareja.
- El OCR del título (`text.cpp`) umbraliza por **percentil global** sobre el
  recorte: con luz no uniforme (un costado lavado, el otro no) el percentil se
  mueve y se come letras. No hay ningún mecanismo de normalización local.
- `badge.cpp` usa umbrales **fijos** `S > 80 && V > 60` sobre el disco. En fotos
  deslavadas u oscuras el disco no los pasa y el histograma de tono vota mal.
  Evidencia: en el detalle de paridad los errores `single → halfdouble` salen
  con `chart_conf` bajo (0.25–0.62), justo al ras del gate `MIN_BADGE_CONF = 0.15`.

### 1.2 Ángulos

- El detector devuelve cajas **axis-aligned**. No hay deskew en ninguna parte
  del pipeline.
- El título se salva gracias a `focusBand` (reencuentra la banda), pero
  **score, nivel y bolita se leen sobre recortes rotados/en perspectiva**:
  glifos pegados, dígitos cortados por el pad, caja de la bolita "de lado".
- El único manejo de distancia es `zoomIn` (segunda pasada dentro del
  `fullscore`), pero opera sobre la foto **rotada**: para la red la pantalla
  diagonal es un objeto distinto del que vio (recto).

### 1.3 Topología de los fallos

El recall del detector con TTA ya es alto:

```
detector recall@0.5 (NCNN vs PyTorch+TTA):
  difficulty 1.0 · fullscore 1.0 · rank 1.0 · score 1.0 · song_name 0.956
```

O sea, las cajas aparecen — el cuello real está en el **OCR sobre luz/ángulo**
y en el **badge**, no en detectar. De las 26 fotos que fallan en el detalle:

- ~8 son `chart_type single → halfdouble` (badge por color, bajo luz/ángulo),
- varias `song: None` o garbage en texto corto (OCR del título),
- `score: None` puntuales (segmentación de dígitos),
- level equivocado que en parte arrastra chart_type.

### 1.4 Infraestructura

- `dataset_v2`, `DATASET`, `venv` de `piu_yolo` y `build_host/` **no están en
  esta máquina** → `tools/parity/parity.py` no corre acá.
- Lo que sí corre local: `gradle testReleaseUnitTest` (ParityTest/GateSweep sobre
  `src/test/resources/parity_fixture.json`, no necesita fotos) y, tras
  `tools/host/fetch.sh`, el binario host `piuocr_cli` sobre imágenes que
  proveamos nosotros.

---

## 2. Estrategia general

1. **Todo opt-in por flag; default intacto.** El `Engine` y el CLI de host ganan
   switches para cada mecanismo nuevo. Sobre las 58 fotos actuales el camino por
   default produce byte-a-byte lo mismo que hoy ⇒ `ParityTest` y el baseline no
   mienten. Cuando la máquina con datos mida un ganador, se flipea el default y
   se regenera `--fixture --update-baseline`.
2. **Medir por condición, no en promedio.** "Mejoró en ángulo pero empeoró en
   luz" se pierde si solo se mira el acierto global: se agregan buckets por foto.
3. **Feedback rápido local sin el dataset**: pantallas PIU **sintéticas**
   dibujadas a mano (ángulo × luz × glare conocidos) dejan validar el código nuevo
   en esta máquina aunque no haya fotos reales.

---

## 3. Fases

### F0 · Infraestructura de medición y flags (primero, sin tocar resultados)

**3.1 Flags en el pipeline**
Archivos: `src/main/cpp/piu_ocr.h`, `src/main/cpp/pipeline.cpp`,
`tools/host/cli.cpp`.

`Engine` gana:

```cpp
struct Options {
  bool rectify = false;           // deskew por perspectiva (F1)
  BinMode binMode = BinMode::Legacy;   // Legacy | Clahe | Adaptive | All (F2)
  BadgeMode badgeMode = BadgeMode::Color; // Color | Fusion (F3)
  int  maxTitleVariants = 1;      // 2 para emitir raw alternativo (F4)
};
```

`cli.cpp` expone `--rectify`, `--bin-mode`, `--badge-mode`, `--title-variants 2`.
Defaults == comportamiento actual.

**3.2 Buckets en `tools/parity/parity.py`**
Clasificar cada foto del detalle por condición y reportar sub-tablas:

- `dark` / `bright` (brillo medio de la foto),
- `glare` (fracción de píxeles casi saturados dentro de la caja `fullscore`),
- `rotated` (cuánto se desvía la caja `fullscore` de recto — aproximado por el
  análisis del panel),
- `far` (área de `fullscore` / área de la foto < 0.4, el mismo umbral de `zoomIn`),
- `twoP` (GT de score con lista de dos valores).

`--buckets` imprime acierto por campo dentro de cada bucket. Se corre en la
máquina con datos (el código queda listo acá).

**3.3 Synthetic smoke local** — `tools/synth.py`
Dibuja pantallas PIU sintéticas (título, bolita con el color del chart, score,
rank, layout Phoenix) y las deforma: perspectiva 0/10/20/30°, brillo 0.4–1.0,
parche de glare gaussiano, ruido. Grabadas en `build/synth/`, corren por
`piuocr_cli` con los flags y se chequea que:

- modo default ⇒ el resultado no cambia vs antes del cambio,
- modo con flag ⇒ lee el ground truth sintético bajo las condiciones dadas.

Sirve de test local donde no hay fotos reales. Requiere una vez
`tools/host/fetch.sh` (baja ncnn + opencv-mobile para Linux).

### F1 · Ángulos — rectify (el lever más grande para song_name en diagonales)

**3.4 `src/main/cpp/rectify.cpp` nuevo + hook en `pipeline.cpp`**

- `estimateQuad(bgr, screenBox)` → 4 esquinas de la pantalla dentro de la caja
  `fullscore`: la pantalla es brillante sobre cabina oscura; binarizar,
  `findContours`, quedarse con el contorno más grande, `minAreaRect` +
  refinar con líneas Hough; validar plausibilidad (área ~ la de la caja,
  aspecto de pantalla). Si no encaja: `false` y el camino actual sigue intacto.
- `warpPerspective` a un rectángulo con la altura de la caja;
- correr ahí dentro la **segunda pasada de detección** (`Engine::zoomIn`
  reemplazado/endurecido), con las cajas mapeadas de vuelta a coords de foto.

La red entonces ve la pantalla axis-aligned — lo mismo que en entrenamiento —
sin retrenar.

`opencv-mobile` ya trae `warpPerspective`, `warpAffine`, `findContours`,
`minAreaRect`, `HoughLinesP` (todos dentro de imgproc/core).

**3.5 Deskew fino del título** (barato, dentro de `text.cpp`/`pipeline.cpp`):
estimar la pendiente de la línea dominante en el ROI y hacer `warpAffine` de
±5° antes de `segmentChars`.

**Validación F1:** smoke sintético con ángulos 0/10/20/30° ⇒ el camino default
sigue igual; con `--rectify` los synth se leen. En la máquina con datos:
`parity.py --detect --baseline --augs` comparando rectificar on/off, y buckets.

### F2 · Luz / glare en el OCR clásico

**3.6 Variantes de binarización en `text.cpp::segmentChars` y
`score.cpp::segmentDigits`:**

- **CLAHE** (`equalizeHist` local — imgproc lo trae),
- **adaptiveThreshold** (media/gaussiana),
- top-hat + percentil originales ya presentes.

Todas puntuadas con el mismo score de coherencia ya existente. En
`binMode=All` se corren todas y gana la mejor; `Legacy` sigue primero para que
las fotos ya bien leídas no cambien.

**Validación F2:** smoke sintético con luz oscura/lavada/glare, y A/B por
buckets (dark, glare) en la máquina con datos.

### F3 · Badge / chart type bajo luz

**3.7 Hue adaptativo en `badge.cpp`:**
calcular los cortes de S/V desde el propio disco (mediana de S, percentil bajo
de V) en vez de `>80 >60` fijos; ponderar el voto por histograma del disco. Un
disco lavado deja de votar "azul" por saturación pobre.

**3.8 Fusión color+texto:**
OCR del sub-label que va sobre los dígitos del nivel dentro de la bolita
("SINGLE"/"DOUBLE"/"HALFDOUBLE"/"COOP") usando `chars.bin` ya embarcado; voto
`color ⊕ texto` (texto decide si legible; si no, color). El badge también se
beneficia del rectify de F1.

**Cascada:** mejor chart_type ⇒ `levelsFor(song, chartType)` en Kotlin acota el
set legal ⇒ level sube solo.

**Validación F3:** A/B `--badge-mode` (Color vs Fusion) en la máquina con datos;
atacar específicamente los errores `single→halfdouble` del detalle (conf ~0.3).

### F4 · Canción lejana / oscura

**3.9 Más cajas y variantes de título:**
`MAX_SONG_BOXES` 3→5 en C++ y Kotlin; el C++ emite por caja la **2ª mejor
lectura** de segmentación en `titles[].raw2` y el pooling de `PiuOcr.interpret`
lo absorbe (hoy ya mezcla candidatos entre 3 cajas).

**3.10 Altura mínima garantizada:** si tras el rectify la caja del título queda
bajita (< ~24 px en la foto), segunda detección con zoom más agresivo;
`segmentChars` normaliza a `TARGET_H=160`, pero con un glifo ya degradado por
un resize previo la interpolación no alcanza.

### F5 · Score extra
El score ya está en 0.905 y su barrido de umbrales es el más sano; solo suma
CLAHE/adaptive como variante (3.6 lo incluye) y el deskew fino de la franja.
Los casos `score: None` del detalle (761729756, 784233147…) pasan por la
binarización de 3.6 y por el rectify de F1 al mejorar la caja.

---

## 4. Validación

### En esta máquina (sin fotos reales)

```bash
gradle testReleaseUnitTest                       # ParityTest/GateSweep sobre fixture
                                                 # (el default nunca cambia ⇒ sigue verde)
tools/host/fetch.sh                              # una sola vez, baja ncnn+opencv-mobile Linux
python3 tools/synth.py --build                   # genera pantallas sintéticas
python3 tools/parity/parity.py --synth           # corre el CLI host sobre lo sintético
                                                 # y valida flags vs default vs GT sintético
```

### En la máquina con datos (dataset_v2 + venv)

```bash
python3 tools/parity/parity.py --detect --baseline                 # referencia
python3 tools/parity/parity.py --detect --augs ... --buckets       # por condición
python3 tools/parity/parity.py --detect --baseline --rectify ...   # A/B por flag
# ganador => flipear default en piu_ocr.h (Options) y:
python3 tools/parity/parity.py --detect --fixture --update-baseline
gradle testReleaseUnitTest
tools/parity/device.sh                             # arm64
```

---

## 5. Riesgos

| riesgo | mitigación |
|---|---|
| Cambiar el default degrada en silencio los 58 GT | todo opt-in; default intacto; flipeo solo tras A/B medido |
| Fixture stale si un flag pasa a default | se regenera con `--fixture --update-baseline` dentro del mismo paso |
| rectify introduce latencia | presupuesto ilimitado; y la pasada va DENTRO de la pantalla (más chica que 1280) |
| CLAHE/adaptiveThreshold agregan candidatos | se puntúan igual que los viejos; Legacy primero ⇒ empate o mejora |
| fetch.sh necesita red local para el binario host | única vez, y sin él solo no corre el smoke sintético (Kotlin tests sí) |

---

## 6. Orden de ejecución

1. F0.1 flags + F0.2 buckets + F0.3 synth (estructura, sin cambiar resultados),
2. F1 rectify (ángulos; mayor lever para song_name),
3. F2 binarización luz (CLAHE/adaptive),
4. F3 badge luz (hue adaptativo + texto),
5. F4 canción (cajas × variantes + altura mínima),
6. medir en máquina con datos, flipear defaults, regenerar fixture+baseline,
7. `device.sh`.

### Archivos que se tocan

| archivo | qué |
|---|---|
| `src/main/cpp/piu_ocr.h` | `Options`, declaraciones de `rectify`, modos |
| `src/main/cpp/detector.cpp` | expone detect sobre imagen ya rectificada |
| `src/main/cpp/rectify.cpp` | **nuevo** estimateQuad / warp |
| `src/main/cpp/pipeline.cpp` | hook rectify (zoomIn 2.0, altura mínima), raw2 |
| `src/main/cpp/text.cpp` | variantes CLAHE/adaptive en `segmentChars`, deskew fino |
| `src/main/cpp/score.cpp` | mismas variantes en `segmentDigits` |
| `src/main/cpp/badge.cpp` | hue adaptativo; fusión color⊕texto |
| `src/main/cpp/jni_bridge.cpp` | pasar Options para runtime (opcional) |
| `tools/host/cli.cpp` | expone flags al host |
| `tools/synth.py` | **nuevo** generador de pantallas |
| `tools/parity/parity.py` | buckets + flags A/B |
| `src/main/kotlin/com/piu/ocr/PiuOcr.kt` | MAX_SONG_BOXES, leer raw2 (si entra) |
| `docs/PLAN_LUZ_ANGULO.md` | este archivo |
