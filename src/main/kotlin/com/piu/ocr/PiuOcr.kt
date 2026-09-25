package com.piu.ocr

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import org.json.JSONObject
import java.io.File
import kotlin.math.sqrt

/**
 * Lectura de una pantalla de resultado de PIU, sin LLM.
 *
 * `reason != null` significa "no confío en esto, escalalo". **Un campo con
 * reason NO se debe escribir en la base.** La política es un nombre si el gate
 * pasa, si no se escala — nunca dos candidatos, porque está medido que debajo
 * del gate el segundo candidato casi nunca acierta.
 */
data class Field<T>(val value: T?, val confidence: Float, val reason: String? = null)

/** Caja en píxeles de la imagen leída. Solo para dibujar sobre la foto; ningún gate la usa. */
data class Box(val x1: Int, val y1: Int, val x2: Int, val y2: Int)

/** Un candidato del cruce con el catálogo, con el puntaje agrupado entre cajas de título. */
data class SongCandidate(val name: String, val score: Double)

data class Reading(
    val song: Field<String>,
    val level: Field<Int>,
    val chartType: Field<String>,
    val score: Field<Int>,
    val rawTitle: String,
    /**
     * Los mejores candidatos del cruce, de mayor a menor. Cuando `song.value` es null por
     * `margen_bajo`, los dos primeros son los que empataron: la política sigue siendo no elegir
     * por ellos; que elija una persona con la foto enfrente.
     */
    val candidates: List<SongCandidate> = emptyList(),
    /** Tamaño de la imagen leída, para normalizar las cajas. 0 si el .so no lo informó. */
    val imageWidth: Int = 0,
    val imageHeight: Int = 0,
    /** Dónde estaba cada cosa. Títulos en el mismo orden que alimentaron [rawTitle]. */
    val titleBoxes: List<Box> = emptyList(),
    val scoreBox: Box? = null,
    val badgeBox: Box? = null,
    val rankBox: Box? = null,
    /** La pantalla de resultado entera (fullscore): adonde la app acerca la foto antes de mostrar. */
    val screenBox: Box? = null,
    /**
     * Grados que el módulo giró la foto para leerla (0, 90 o 270), cuando la pantalla estaba de
     * costado y el detector no la veía. Las cajas YA vienen en coordenadas de la foto original:
     * esto es para que la app muestre la pantalla derecha en la animación, sin re-mapear nada.
     */
    val turnedBy: Int = 0,
) {
    val needsLlm: List<String> get() = buildList {
        if (song.reason != null || song.value == null) add("song")
        if (level.reason != null || level.value == null) add("level")
        if (chartType.reason != null || chartType.value == null) add("chart_type")
        if (score.reason != null || score.value == null) add("score")
    }
    /** Una llamada al VLM devuelve todos los campos: el costo es por pantalla,
     *  no por campo. Alcanza con que uno no pase para pagarla entera. */
    val needsVlmCall: Boolean get() = needsLlm.isNotEmpty()
}

class PiuOcr private constructor(
    handle: Long,
    internal val matcher: SongMatcher,
) : AutoCloseable {

    @Volatile private var handle = handle

    fun read(bitmap: Bitmap): Reading = interpret(readRaw(bitmap), matcher)

    /** JSON crudo del .so. Lo usa el test en device (tools/parity/device.sh)
     *  para comparar contra el CLI de host foto por foto. */
    internal fun readRaw(bitmap: Bitmap): String {
        // El .so solo lee ARGB_8888 con píxeles bloqueables. Un HARDWARE bitmap
        // (lo que devuelve ImageDecoder por defecto en API 28+) o un RGB_565
        // fallaban en lockPixels y volvían vacíos en silencio.
        val bmp = if (bitmap.config == Bitmap.Config.ARGB_8888 && !isHardware(bitmap)) bitmap
                  else bitmap.copy(Bitmap.Config.ARGB_8888, false)
        try {
            // Mismo lock que close(): sin esto, cerrar en otro hilo entre el
            // check y nativeRead era use-after-free en el .so.
            return synchronized(this) {
                check(handle != 0L) { "PiuOcr ya está cerrado" }
                nativeRead(handle, bmp)
            }
        } finally {
            // La copia es nuestra: esperar al GC eran ~12 MB por lectura de un
            // bitmap HARDWARE.
            if (bmp !== bitmap) bmp?.recycle()
        }
    }

    /**
     * Activa mecanismos opt-in (default == comportamiento actual, ver
     * docs/PLAN_LUZ_ANGULO.md): rectify (F1), binMode (F2), badgeMode (F3),
     * titleVariants (F4). Los modos numéricos siguen el orden del enum de C++.
     */
    fun setOptions(
        rectify: Boolean = false,
        binMode: Int = BIN_LEGACY,
        badgeMode: Int = BADGE_COLOR,
        titleVariants: Int = 1,
        titleBoxes: Int = DEFAULT_TITLE_BOXES,
    ) {
        synchronized(this) {
            check(handle != 0L) { "PiuOcr ya está cerrado" }
            nativeSetOptions(handle, rectify, binMode, badgeMode, titleVariants, titleBoxes)
        }
    }

    /** Idempotente: un segundo close() era un double free en el .so. */
    @Synchronized
    override fun close() {
        if (handle != 0L) { nativeDestroy(handle); handle = 0L }
    }

    private fun isHardware(b: Bitmap) =
        Build.VERSION.SDK_INT >= 26 && b.config == Bitmap.Config.HARDWARE

    companion object {
        private var loaded = false

    /**
     * Interpreta el JSON del .so: gates y cruce con el catálogo. Es una
     * función pura (sin Context ni Bitmap) para que el test de paridad
     * (tools/parity) la corra en la JVM sobre salidas grabadas del CLI.
     */
    @JvmStatic
    internal fun interpret(nativeJson: String, matcher: SongMatcher): Reading {
        val j = JSONObject(nativeJson)

        // chart_type primero: acota qué canciones son posibles.
        val ctRaw = j.optString("chart_type").ifEmpty { null }
        val ctConf = j.optDouble("chart_conf", 0.0).toFloat()
        val chart = Field(
            if (ctConf >= MIN_BADGE_CONF) ctRaw else null, ctConf,
            if (ctRaw != null && ctConf >= MIN_BADGE_CONF) null else "baja_confianza")

        // Una pantalla 2P imprime el título dos veces y los dos recortes fallan
        // distinto (reflejo, recorte), así que se juntan los puntajes: una
        // lectura limpia le gana a una colapsada. Vale +7 pts, medido.
        val titles = j.optJSONArray("titles") ?: org.json.JSONArray()
        val pooled = HashMap<String, Double>()
        val raws = ArrayList<String>()
        val titleBoxes = ArrayList<Box>()
        for (i in 0 until minOf(titles.length(), MAX_SONG_BOXES)) {
            val t = titles.getJSONObject(i)
            val raw = t.getString("raw")
            if (raw.isEmpty()) continue
            raws += raw
            t.optBox("box")?.let { titleBoxes += it }
            // Ponderar por la confianza de la caja: una caja de 0.005 puede ser
            // la respuesta cuando es la única, sin ganarle a una de 0.5.
            val w = sqrt(maxOf(t.optDouble("conf", 1.0), 0.02))
            // F4.9: la 2ª mejor lectura de la misma caja (raw2) entra al pool con
            // peso algo menor; solo existe si el C++ corrió con --title-variants 2.
            // Va también a `raws` para que el gate `looksLike` lo vea: si el
            // ganador salió de raw2, rechazarlo por "sin_parecido" era un falso null.
            val raw2 = t.optString("raw2").takeIf { it.isNotEmpty() && it != raw }
            if (raw2 != null) raws += raw2
            val texts = if (raw2 != null) listOf(raw, raw2) else listOf(raw)
            for ((ti, text) in texts.withIndex()) {
                val tw = if (ti == 0) w else w * 0.85
                for (c in matcher.match(text, chart.value, topK = 8)) {
                    val e = pooled[c.name] ?: 0.0
                    pooled[c.name] = maxOf(e, c.score * tw) + 0.15 * minOf(e, c.score * tw)
                }
            }
        }
        val ranked = pooled.entries.sortedByDescending { it.value }
        val margin = if (ranked.size > 1) ranked[0].value - ranked[1].value else 1.0
        // Dos gates. El margen dice que el pool está decidido; el parecido dice que lo que ganó
        // se parece a lo LEÍDO. Sin el segundo, el pool ordena bien y separa mal: los ocho
        // errores del fixture eran nombres cortos (Bee, N, See, Point Break) que ganan sobre
        // glifos basura porque todo se parece un poco a "N".
        val clear = ranked.isNotEmpty() && margin >= MIN_SONG_MARGIN
        val alike = clear && looksLike(raws, ranked[0].key)
        val song = Field(if (alike) ranked[0].key else null, margin.toFloat(),
                         when { alike -> null; clear -> "sin_parecido"; else -> "margen_bajo" })

        // Con la canción resuelta el catálogo dice qué niveles son legales, y
        // eso reordena los dígitos leídos en vez de solo aceptar el argmax.
        val level = readLevel(j, song.value, chart.value, matcher)
        return Reading(
            song, level, chart, readScore(j), raws.joinToString(" | "),
            candidates = ranked.take(MAX_CANDIDATES).map { SongCandidate(it.key, it.value) },
            imageWidth = j.optInt("w", 0),
            imageHeight = j.optInt("h", 0),
            titleBoxes = titleBoxes,
            scoreBox = j.optBox("score_box"),
            badgeBox = j.optBox("badge_box"),
            rankBox = j.optBox("rank_box"),
            screenBox = j.optBox("screen_box"),
            turnedBy = j.optInt("turned_by", 0),
        )
    }

    /**
     * ¿Alguna de las lecturas crudas se parece al nombre que ganó? difflib ratio sobre lo
     * normalizado, sin espacios (el mismo `similarity` del matcher).
     *
     * Medido sobre las 83 fotos con verdad del fixture, con MIN_SONG_MARGIN 0.010:
     *   sin este gate        66 bien /  8 mal /  9 null
     *   piso 0.50            66 bien /  4 mal / 13 null
     *   piso 0.55            63 bien /  2 mal / 18 null   <- este
     *   piso 0.70            50 bien /  0 mal / 33 null
     * Los dos que quedan a 0.55 son nombres de 1 y 3 letras ("N", "Bee") que cualquier
     * basura roza; por eso un nombre corto exige casi coincidencia exacta.
     */
    private fun looksLike(raws: List<String>, name: String): Boolean {
        val n = SongMatcher.normalize(name).replace(" ", "")
        if (n.isEmpty()) return false
        // Un raw más corto que el piso (relativo al largo del nombre) no
        // sostiene el match: 'o-C' -> 'BOCA' era un falso positivo con conf de
        // detector 0.88; 'B3' con raw exacto 'B3' sobrevive porque el piso es
        // relativo. Ver threshold_sweep.py y ARQUITECTURA §8.
        val floor = minOf(MIN_RAW_LEN, n.length)
        val need = if (n.length <= SHORT_NAME) MIN_RAW_SIMILARITY_SHORT else MIN_RAW_SIMILARITY
        return raws.any { raw ->
            val r = SongMatcher.normalize(raw).replace(" ", "")
            r.isNotEmpty() && r.length >= floor && SongMatcher.similarity(r, n) >= need
        }
    }

    private fun JSONObject.optBox(key: String): Box? {
        val a = optJSONArray(key) ?: return null
        if (a.length() != 4) return null
        return Box(a.getInt(0), a.getInt(1), a.getInt(2), a.getInt(3))
    }

    /**
     * Score, ya validado contra el dominio en C++ (0..1000000, y el "cero"
     * gris de la izquierda descartado). Acá solo queda el gate por margen.
     */
    private fun readScore(j: JSONObject): Field<Int> {
        val v = j.optInt("score", -1)
        if (v < 0) return Field(null, 0f, "sin_glifos")
        val m = j.optDouble("score_margin", 0.0).toFloat()
        if (m >= MIN_SCORE_MARGIN) return Field(v, m, null)
        // Acuerdo: una segunda lectura con otro recorte (ver pipeline.cpp) dio el mismo número.
        val agree = j.optInt("score2", -2) == v
        return if (agree && m >= MIN_SCORE_AGREE_MARGIN) Field(v, m, null)
               else Field(null, m, "margen_bajo")
    }

    private fun readLevel(j: JSONObject, song: String?, chartType: String?,
                          matcher: SongMatcher): Field<Int> {
        val sc = j.optJSONArray("level_scores") ?: return Field(null, 0f, "sin_glifos")
        if (sc.length() == 0) return Field(null, 0f, "sin_glifos")
        // La bolita dibuja siempre dos dígitos ("04"): un nivel legal de un dígito se compara con
        // cero a la izquierda. Antes se filtraba por longitud y un 4 nunca podía ganar.
        val legal = song?.let { matcher.levelsFor(it, chartType) }.orEmpty()
            .filter { it.toString().length <= sc.length() }
        if (legal.isEmpty()) {
            val d = j.optJSONArray("level_digits") ?: return Field(null, 0f, "sin_glifos")
            val v = (0 until d.length()).joinToString("") { d.getInt(it).toString() }
                .toIntOrNull() ?: return Field(null, 0f, "no_numerico")
            // value null cuando hay reason, como en canción y chart_type: un
            // 81 con reason="fuera_de_rango" igual terminaba en la base.
            return if (v in 1..28) Field(v, 0f, null) else Field(null, 0f, "fuera_de_rango")
        }
        val scored = legal.map { cand ->
            cand to cand.toString().padStart(sc.length(), '0').withIndex().sumOf { (i, ch) ->
                sc.getJSONArray(i).optDouble(ch - '0', -1.0)
            }
        }.sortedByDescending { it.second }
        val m = if (scored.size > 1)
            (scored[0].second - scored[1].second) / sc.length() else 1.0
        return Field(scored[0].first, m.toFloat(), null)
    }


        /**
         * Carga la librería nativa.
         *
         * Si este módulo va como **dynamic feature de Play**, un `.so` adentro
         * NO se carga con `System.loadLibrary`: hay que pasar por
         * `SplitCompat.install()` + `SplitInstallHelper.loadLibrary()`. Y falla
         * SOLO en builds firmados de Play — nunca en debug ni en un APK local —
         * así que se descubre en producción si no se contempla.
         *
         * Se resuelve por reflexión para no obligar a la dependencia de Play
         * Core cuando el módulo va embebido normal.
         */
        @Synchronized
        private fun ensureLoaded(context: Context) {
            if (loaded) return
            try {
                val helper = Class.forName(
                    "com.google.android.play.core.splitcompat.SplitInstallHelper")
                helper.getMethod("loadLibrary", Context::class.java, String::class.java)
                    .invoke(null, context, "piuocr")
            } catch (_: Throwable) {
                System.loadLibrary("piuocr")     // módulo embebido normal
            }
            loaded = true
        }

        // Gates medidos end-to-end con LOSO por foto. Ver docs/ARQUITECTURA.md §8:
        // cambiarlos degrada el sistema EN SILENCIO.
        //   canción  0.001 -> cob 0.778 / prec 1.000   <- este (2026-09-19)
        //            0.010 -> cob 0.889 / prec 0.900   (2026-09-09)
        //            0.015 -> cob 0.800 / prec 0.972
        //            0.030 -> cob 0.756 / prec 0.971
        // Bajado de 0.010 a 0.001: recupera "Super Capriccio" (margen 0.0017) sin
        // agregar errores ni en device (63 fotos) ni en el fixture (90). `looksLike`
        // sigue siendo el gate que ataja las lecturas basura.
        const val MIN_SONG_MARGIN = 0.001
        // Gate del score, medido con LOSO sobre 52 fotos con score leído a mano
        // (ver pipeline.py MIN_SCORE_MARGIN_SAFE):
        //   0.020 -> cob 0.79 / prec 1.00
        //   0.010 -> cob 0.94 / prec 0.98
        //   0.003 -> cob 0.94 / prec 0.88   (2026-09-19, device v4: 63 fotos)
        //   0.000 -> cob 1.00 / prec 0.83
        // 2026-09-25, con la SEGUNDA lectura del score (pipeline.cpp) y 115 scores con GT
        // (63 nuevas + viejas), nativo host con yolo26_v5d:
        //   0.003 sin acuerdo        -> 0.835 bien / 9.6 % errores aceptados
        //   0.010 sin acuerdo        -> 0.774 / 4.3 %
        //   0.010 + acuerdo (≥ 0.0)  -> 0.861 / 6.1 %   <- este: gana en las dos
        const val MIN_SCORE_MARGIN = 0.010f
        // Por debajo de MIN_SCORE_MARGIN se acepta igual si las dos lecturas del score coinciden
        // (recortes distintos, mismo número) y el margen llega a este piso.
        const val MIN_SCORE_AGREE_MARGIN = 0.0f
        // 0.35 no compraba precisión, solo la tiraba: sobre 55 bolitas 0.15 da
        // 0.873 y 0.35 da 0.655, porque convierte lecturas buenas en null.
        const val MIN_BADGE_CONF = 0.15f
        // F4.9: 3 -> 5 cajas de título. El C++ por default emite 3 (Options
        // maxTitleBoxes), así que el fixture y el resultado no cambian hasta
        // que se corra con el flag; este tope solo deja de recortar de más.
        const val MAX_SONG_BOXES = 5
        /** Cajas que emite el C++ sin setOptions: el default no se toca. */
        const val DEFAULT_TITLE_BOXES = 3
        /** Candidatos que viajan en [Reading.candidates]: los que empatan y uno más de contexto. */
        const val MAX_CANDIDATES = 3

        /** Parecido mínimo (difflib ratio) entre alguna lectura cruda y el nombre que ganó. Ver looksLike. */
        const val MIN_RAW_SIMILARITY = 0.55
        /** Un nombre de hasta [SHORT_NAME] caracteres exige casi coincidencia: "N" se parece a todo. */
        const val MIN_RAW_SIMILARITY_SHORT = 0.80
        const val SHORT_NAME = 3
        /** Piso de caracteres (normalizado, sin espacios) de un raw para sostener
         * un match, relativo al largo del nombre: floor = min(MIN_RAW_LEN, len(nombre)).
         * 'o-C' -> 'BOCA' era un falso positivo con conf de detector 0.88; 'B3' con
         * raw exacto 'B3' sobrevive. Medido en threshold_sweep.py: mata 1 FP de song
         * (4->3 en device 63) sin perder lecturas correctas (fixture 90 intacto). */
        const val MIN_RAW_LEN = 4

        // digits.bin ES obligatorio: Engine::load lo exige y sin él create() tiraba
        // para todo el mundo. Faltaba en esta lista desde que se agregó el score.
        // chart_knn.bin: banco del kNN de tipo de chart (opcional en C++; sin él vuelve a los
        // rangos de tono). Tiene que estar en la lista o no se copia del APK al disco.
        private val ASSETS = listOf("chars.bin", "level.bin", "digits.bin", "catalog.json",
                                    "chart_knn.bin", "piu_yolo.param", "piu_yolo.bin")

        /**
         * Copia los assets a filesDir y crea el lector. Falla con
         * [IllegalStateException] si el .so no pudo cargar los modelos: antes
         * devolvía un handle a medias que leía vacío para siempre.
         *
         * La copia se hace una vez POR VERSIÓN de la app: si solo se mira
         * `exists()`, una actualización del APK con modelos nuevos sigue usando
         * los viejos. Y cada archivo se escribe a `.tmp` y se renombra, para que
         * un crash a mitad de copia no deje un `.bin` truncado que "existe".
         */
        /** Sincronizado: dos create() a la vez copiaban los assets a los
         *  mismos `.tmp` y podían dejar un modelo truncado que "existe". */
        @JvmStatic
        @Synchronized
        fun create(context: Context): PiuOcr {
            val dir = File(context.filesDir, "piu_ocr").apply { mkdirs() }
            val stamp = File(dir, ".version")
            val want = appVersion(context)
            if (stamp.takeIf { it.exists() }?.readText() != want) {
                for (n in ASSETS) {
                    val tmp = File(dir, "$n.tmp")
                    context.assets.open("piu_ocr/$n").use { i ->
                        tmp.outputStream().use { o -> i.copyTo(o) }
                    }
                    val dst = File(dir, n)
                    if (!tmp.renameTo(dst)) { dst.delete(); check(tmp.renameTo(dst)) }
                }
                stamp.writeText(want)
            }
            ensureLoaded(context)
            val h = nativeCreate(dir.absolutePath)
            check(h != 0L) { "piuocr: no se pudieron cargar los modelos de ${dir.absolutePath}" }
            return PiuOcr(h, SongMatcher(File(dir, "catalog.json").readText()))
        }

        private fun appVersion(context: Context): String {
            val pi = context.packageManager.getPackageInfo(context.packageName, 0)
            val code = if (Build.VERSION.SDK_INT >= 28) pi.longVersionCode
                       else @Suppress("DEPRECATION") pi.versionCode.toLong()
            return "$code:${pi.lastUpdateTime}"
        }

        // Modos opt-in (F0.1). Ver Options en src/main/cpp/piu_ocr.h.
        const val BIN_LEGACY = 0
        const val BIN_CLAHE = 1
        const val BIN_ADAPTIVE = 2
        const val BIN_ALL = 3
        const val BADGE_COLOR = 0
        const val BADGE_ADAPTIVE = 1
        const val BADGE_FUSION = 2

        @JvmStatic private external fun nativeCreate(assetDir: String): Long
        @JvmStatic private external fun nativeDestroy(handle: Long)
        @JvmStatic private external fun nativeRead(handle: Long, bitmap: Bitmap): String
        @JvmStatic private external fun nativeSetOptions(
            handle: Long, rectify: Boolean, binMode: Int, badgeMode: Int,
            titleVariants: Int, titleBoxes: Int)
    }
}
