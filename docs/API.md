# API

El módulo expone una sola clase pública, `com.piu.ocr.PiuOcr`. Todo lo demás
(`Reading`, `Field`, `Box`, `SongCandidate`, `SongMatcher`) son tipos de apoyo.

```kotlin
val ocr = PiuOcr.create(context)
val r = ocr.read(bitmap)

r.score.value          // Int?          el puntaje leído
r.score.confidence     // Float         0..1
r.score.reason         // String?       null si se confía en el valor

if (r.needsVlmCall) escalarAlVlm(r)   // no confiar en ningún reason != null
ocr.close()
```

Requisitos del módulo: `minSdk 24`. El AAR trae las cuatro ABIs
(`arm64-v8a`, `armeabi-v7a`, `x86`, `x86_64`).

## Integración

### Como módulo del mismo proyecto

```kotlin
// settings.gradle.kts
include(":piu-ocr")

// app/build.gradle.kts
implementation(project(":piu-ocr"))
```

### Como AAR

```kotlin
// app/build.gradle.kts
implementation(files("libs/piu-ocr-release.aar"))
```

No hace falta ninguna dependencia transitiva: el `.so` linkea ncnn y
opencv-mobile de forma estática.

> **Dynamic feature de Play:** si el módulo se descarga por separado, un `.so`
> adentro **no** carga con `System.loadLibrary`. El módulo lo resuelve por
> reflexión (`SplitInstallHelper`), así que no hay que agregar Play Core; pero
> conviene saberlo porque falla **solo en builds firmados de Play**.

## `PiuOcr`

```kotlin
class PiuOcr : AutoCloseable {
    fun read(bitmap: Bitmap): Reading
    fun setOptions(
        rectify: Boolean = false,
        binMode: Int = BIN_LEGACY,
        badgeMode: Int = BADGE_COLOR,
        titleVariants: Int = 1,
        titleBoxes: Int = DEFAULT_TITLE_BOXES,
    )
    override fun close()
}
```

### `create(context): PiuOcr`

Copia los assets a `filesDir/piu_ocr` (una vez **por versión** de la app),
carga la librería nativa y crea el lector. Falla con `IllegalStateException` si
el `.so` no pudo cargar los modelos — antes devolvía un handle a medias que leía
vacío para siempre.

### `read(bitmap): Reading`

Lee una pantalla de resultado. Internamente copia el bitmap a `ARGB_8888` si
hace falta (un `HARDWARE` de `ImageDecoder` o un `RGB_565` fallaban en silencio).

Es una llamada pesada (detector + OCR). Conviene correrla fuera del hilo
principal; **no** está documentada como thread-safe, así que no la llames en
paralelo sobre la misma instancia.

### `setOptions(...)`

Activa mecanismos opt-in. El **default reproduce el comportamiento actual**;
usá esto solo si estás midiendo un A/B (ver
[PLAN_LUZ_ANGULO.md](PLAN_LUZ_ANGULO.md)).

| parámetro | constantes | qué hace |
|---|---|---|
| `rectify` | `true`/`false` | endereza la pantalla por perspectiva |
| `binMode` | `BIN_LEGACY`, `BIN_CLAHE`, `BIN_ADAPTIVE`, `BIN_ALL` | binarización del OCR |
| `badgeMode` | `BADGE_COLOR`, `BADGE_ADAPTIVE`, `BADGE_FUSION` | voto del chart type |
| `titleVariants` | 1 o 2 | 2 emite una segunda lectura del título por caja |
| `titleBoxes` | 3 por defecto | máximo de cajas de título a considerar |

### `close()`

Idempotente: un segundo `close()` era un double free en el `.so`. Usalo con
`use { }`.

## `Reading`

```kotlin
data class Reading(
    val song: Field<String>,
    val level: Field<Int>,
    val chartType: Field<String>,
    val score: Field<Int>,
    val rawTitle: String,
    val candidates: List<SongCandidate> = emptyList(),
    val imageWidth: Int = 0,
    val imageHeight: Int = 0,
    val titleBoxes: List<Box> = emptyList(),
    val scoreBox: Box? = null,
    val badgeBox: Box? = null,
    val rankBox: Box? = null,
    val screenBox: Box? = null,
    val turnedBy: Int = 0,
) {
    val needsLlm: List<String>
    val needsVlmCall: Boolean
}
```

### `Field<T>`

```kotlin
data class Field<T>(val value: T?, val confidence: Float, val reason: String? = null)
```

`reason != null` significa **"no confío en esto, escalalo"**. Un campo con
`reason` **no se debe escribir en la base**. La política es un nombre si el gate
pasa, si no se escala — nunca dos candidatos, porque está medido que debajo del
gate el segundo candidato casi nunca acierta.

Valores de `reason`:

| campo | `reason` | significado |
|---|---|---|
| song | `margen_bajo` | el 1.º y el 2.º candidato del catálogo están muy cerca |
| song | `sin_parecido` | algo ganó el pool pero no se parece a lo leído |
| chart_type | `baja_confianza` | el color de la bolita no fue concluyente |
| score | `margen_bajo` | el clasificador no distingue entre dígitos |
| score | `sin_glifos` | no se segmentó ningún dígito |
| level | `sin_glifos` / `no_numerico` / `fuera_de_rango` | no hay lectura legal |

### `needsLlm` y `needsVlmCall`

```kotlin
val needsLlm: List<String>   // campos en los que no se confía
val needsVlmCall: Boolean    // alguno de los anteriores
```

Una llamada al VLM devuelve todos los campos: el costo es **por pantalla**, no
por campo. Alcanza con que uno no pase el gate para pagarla entera.

### `candidates`

Los mejores candidatos del cruce con el catálogo, de mayor a menor. Cuando
`song.value` es `null` por `margen_bajo`, los dos primeros son los que
empataron: la política sigue siendo **no elegir por ellos**; que elija una
persona con la foto enfrente.

### Cajas

`titleBoxes`, `scoreBox`, `badgeBox`, `rankBox` y `screenBox` vienen en píxeles
de la imagen y sirven **solo para dibujar** sobre la foto (la app anima dónde
vio cada cosa). `screenBox` es la pantalla de resultado entera: el lugar adonde
conviene acercar la foto. Ningún gate las usa. `imageWidth`/`imageHeight` son 0
si el `.so` no los informó, para normalizar las cajas.

`turnedBy` son los grados que el módulo giró la foto para leerla (0, 90 o 270),
cuando la pantalla estaba de costado y el detector no la veía. Las cajas **ya
vienen en coordenadas de la foto original**, así que `turnedBy` es solo para que
la app muestre la pantalla derecha en la animación (rotando foto y cajas
juntas), sin re-mapear nada.

## `SongMatcher`

`SongMatcher(catalogJson)` hace el matching difuso contra el catálogo cerrado
(675 canciones). Es público pero normalmente no se instancia a mano: `create()`
ya deja uno listo en `PiuOcr.matcher`. Está documentado en
[ARQUITECTURA.md §8](ARQUITECTURA.md#8-gates-kotlin).

## Ejemplo completo

```kotlin
class ResultActivity : AppCompatActivity() {
    private lateinit var ocr: PiuOcr

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ocr = PiuOcr.create(this)
    }

    private fun onPhoto(bitmap: Bitmap) = lifecycleScope.launch(Dispatchers.Default) {
        val r = ocr.read(bitmap)
        if (r.needsVlmCall) {
            // La foto es la fuente de verdad: mostrar la lectura antes de guardar.
            showForReview(r)
        } else {
            save(r.song.value!!, r.level.value!!, r.chartType.value!!, r.score.value!!)
        }
    }

    override fun onDestroy() {
        ocr.close()
        super.onDestroy()
    }
}
```
