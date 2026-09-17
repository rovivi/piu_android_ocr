package com.piu.ocr

import org.json.JSONObject
import java.text.Normalizer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

/**
 * Matching difuso del título contra el catálogo cerrado de 675 canciones.
 *
 * Vive en Kotlin y no en C++ a propósito: es manipulación de strings, no es el
 * cuello de latencia, y evita reimplementar SequenceMatcher en C++.
 *
 * El OCR crudo es feísimo — `waodingGrashorg`, `DDATTUGnux` — y aun así el
 * catálogo cerrado lo rescata. No hace falta leer bien, hace falta leer parecido.
 */
class SongMatcher(catalogJson: String) {

    data class Song(val name: String, val charts: List<Pair<String, Int>>) {
        val norm = normalize(name)
        val trigrams = trigramsOf(norm)
        val compact = norm.replace(" ", "")
    }

    data class Candidate(val name: String, val score: Double, val levels: List<Int>)

    // En el ORDEN del archivo, como el dict de Python. JSONObject.keys() no lo
    // garantiza (HashMap en la JVM, LinkedHashMap en Android) y el orden decide
    // los empates del prefiltro y del ranking: ParityTest lo detectó.
    private val songs: List<Song> = JSONObject(catalogJson).let { root ->
        topLevelKeys(catalogJson).asSequence().filter { !it.startsWith("_") }.map { k ->
            val o = root.getJSONObject(k)
            val ch = o.optJSONArray("c")
            val charts = buildList {
                for (i in 0 until (ch?.length() ?: 0)) {
                    val c = ch!!.getJSONArray(i)
                    if (!c.isNull(1)) add(c.getString(0) to c.getInt(1))
                }
            }
            Song(o.getString("n"), charts)
        }.toList()
    }

    /**
     * @param chartType single/double/halfdouble/coop, o null.
     *   El NIVEL a propósito no es parámetro: `match` lo usaría como filtro duro
     *   y el nivel que tenemos en inferencia es LEÍDO, no sabido. Un nivel mal
     *   leído borra la canción correcta. Medido: el cruce con el nivel vale
     *   +3 pts cuando el nivel es correcto y es pérdida neta cuando no.
     */
    fun match(text: String, chartType: String? = null, topK: Int = 2): List<Candidate> {
        val q = normalize(text)
        if (q.isEmpty()) return emptyList()
        val qt = trigramsOf(q)
        val qc = q.replace(" ", "")

        // chartType NO filtra el pool: en Catalog.match de Python el step type
        // solo actúa junto con el nivel (_charts_allow devuelve True sin
        // nivel), y el nivel nunca se pasa en inferencia. Filtrar acá sacaba
        // candidatos del top-8 y movía el margen del gate (ParityTest).
        var pool = songs

        // difflib es el 90 % del costo del matching (dos pasadas por canción,
        // 675 canciones). El Jaccard de trigramas es aritmética de conjuntos y
        // ordena la misma región del catálogo, así que se puntúa todo con eso y
        // solo se paga la parte cara sobre la cabeza. Medido, 60/120/240 y sin
        // prefiltro dan el mismo top-1.
        if (pool.size > PREFILTER) {
            pool = pool.sortedByDescending { jaccard(qt, it.trigrams) }.take(PREFILTER)
        }

        return pool.map { s ->
            val tri = jaccard(qt, s.trigrams)
            val ratio = similarity(qc, s.compact)
            val partial = longestCommon(qc, s.compact).toDouble() /
                max(min(qc.length, s.compact.length), 1)
            // round(sc, 4) como en Catalog.match: el gate de margen (0.015)
            // se decide en el cuarto decimal y sin esto Kotlin y Python
            // divergían en 3 de 90 fotos (ParityTest).
            Candidate(s.name, round((0.45 * tri + 0.35 * ratio + 0.20 * partial) * 1e4) / 1e4,
                      s.charts.map { it.second }.distinct().sorted())
        }.sortedByDescending { it.score }.take(topK)
    }

    /** Niveles que el catálogo permite para esa canción y ese step type. */
    fun levelsFor(name: String, chartType: String?): List<Int> {
        val s = songs.firstOrNull { it.name == name || it.norm == normalize(name) }
            ?: return emptyList()
        val want = chartKey(chartType)
        val lv = s.charts.filter { want == null || it.first == want || it.first.isEmpty() }
            .map { it.second }.distinct().sorted()
        return lv.ifEmpty { s.charts.map { it.second }.distinct().sorted() }
    }

    companion object {
        /** Claves de primer nivel de un objeto JSON, en orden de aparición.
         *  Escáner mínimo: sigue la profundidad y respeta escapes en strings. */
        internal fun topLevelKeys(json: String): List<String> {
            val out = ArrayList<String>()
            var depth = 0; var i = 0
            while (i < json.length) {
                when (val c = json[i]) {
                    '{', '[' -> depth++
                    '}', ']' -> depth--
                    '"' -> {
                        val sb = StringBuilder(); i++
                        while (i < json.length && json[i] != '"') {
                            if (json[i] == '\\' && i + 1 < json.length) {
                                i++
                                when (json[i]) {
                                    'n' -> sb.append('\n'); 't' -> sb.append('\t')
                                    'u' -> { sb.append(json.substring(i + 1, i + 5).toInt(16).toChar()); i += 4 }
                                    else -> sb.append(json[i])
                                }
                            } else sb.append(json[i])
                            i++
                        }
                        if (depth == 1) {
                            var k = i + 1
                            while (k < json.length && json[k].isWhitespace()) k++
                            if (k < json.length && json[k] == ':') out += sb.toString()
                        }
                    }
                    else -> {}
                }
                i++
            }
            return out
        }

        private fun chartKey(chartType: String?) = when (chartType) {
            "single" -> "s"; "double" -> "d"; "halfdouble" -> "hd"; "coop" -> "c"
            else -> null
        }

        /** Holgado respecto de los 60 que ya bastaban: el margen del gate
         *  depende del segundo candidato, no solo del primero. */
        const val PREFILTER = 120

        // Precompiladas: normalize corre 675 veces al cargar el catálogo y en
        // cada match; Regex(...) inline compilaba el patrón cada vez.
        private val NON_ALNUM = Regex("[^a-z0-9\\s]")
        private val SPACES = Regex("\\s+")

        private val MARKS = Regex("\\p{Mn}+")

        /** Igual a song_match.normalize: NFD, sin diacríticos, minúsculas,
         *  todo lo que no sea [a-z0-9] es espacio. Sin el NFD, "é" se volvía
         *  espacio acá y "e" en Python. */
        fun normalize(s: String): String =
            Normalizer.normalize(s, Normalizer.Form.NFD).replace(MARKS, "").lowercase()
                .replace(NON_ALNUM, " ")
                .replace(SPACES, " ").trim()

        fun trigramsOf(n: String): Set<String> {
            val s = "  ${n.replace(" ", "")}  "
            return if (s.length >= 3) (0..s.length - 3).map { s.substring(it, it + 3) }.toSet()
                   else setOf(s)
        }

        private fun jaccard(a: Set<String>, b: Set<String>): Double {
            // Iterar el chico y buscar en el grande.
            val inter = if (a.size <= b.size) a.count { it in b } else b.count { it in a }
            val union = a.size + b.size - inter
            return if (union > 0) inter.toDouble() / union else 0.0
        }

        /**
         * SequenceMatcher.ratio() de difflib, porteado tal cual. NO es LCS:
         * difflib toma el bloque común más largo, recursa a izquierda y
         * derecha, y suma. Para `abcXdef` vs `defXabc` LCS da 4 y difflib da 3.
         * La versión anterior usaba LCS y el gate de canción divergía de
         * Python en fotos al borde del margen.
         */
        internal fun similarity(a: String, b: String): Double {
            if (a.isEmpty() && b.isEmpty()) return 1.0
            val m = matchingBlocksSize(a, b)
            return 2.0 * m / (a.length + b.length)
        }

        private class Match(val i: Int, val j: Int, val size: Int)

        /** find_longest_match sin junk (b < 200 chars: autojunk no aplica). */
        private fun longestMatch(a: String, b: String, b2j: Map<Char, IntArray>,
                                 alo: Int, ahi: Int, blo: Int, bhi: Int): Match {
            var besti = alo; var bestj = blo; var bestsize = 0
            var j2len = IntArray(b.length + 1)          // índice j+1
            var newj2len = IntArray(b.length + 1)
            for (i in alo until ahi) {
                java.util.Arrays.fill(newj2len, 0)
                val js = b2j[a[i]]
                if (js != null) for (j in js) {
                    if (j < blo) continue
                    if (j >= bhi) break
                    val k = j2len[j] + 1                  // j2len.get(j-1, 0) + 1
                    newj2len[j + 1] = k
                    if (k > bestsize) { besti = i - k + 1; bestj = j - k + 1; bestsize = k }
                }
                val t = j2len; j2len = newj2len; newj2len = t
            }
            while (besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1]) {
                besti--; bestj--; bestsize++
            }
            while (besti + bestsize < ahi && bestj + bestsize < bhi &&
                   a[besti + bestsize] == b[bestj + bestsize]) bestsize++
            return Match(besti, bestj, bestsize)
        }

        /** Suma de tamaños de get_matching_blocks (el merge de adyacentes no
         *  cambia la suma). */
        private fun matchingBlocksSize(a: String, b: String): Int {
            val b2j = HashMap<Char, ArrayList<Int>>()
            for ((j, c) in b.withIndex()) b2j.getOrPut(c) { ArrayList() }.add(j)
            val idx = b2j.mapValues { it.value.toIntArray() }
            val queue = ArrayDeque<IntArray>()
            queue.addLast(intArrayOf(0, a.length, 0, b.length))
            var total = 0
            while (queue.isNotEmpty()) {
                val (alo, ahi, blo, bhi) = queue.removeLast()
                val m = longestMatch(a, b, idx, alo, ahi, blo, bhi)
                if (m.size > 0) {
                    total += m.size
                    if (alo < m.i && blo < m.j) queue.addLast(intArrayOf(alo, m.i, blo, m.j))
                    if (m.i + m.size < ahi && m.j + m.size < bhi)
                        queue.addLast(intArrayOf(m.i + m.size, ahi, m.j + m.size, bhi))
                }
            }
            return total
        }

        /** Corrida común más larga: un recorte de título viene cortado por el
         *  borde del cuadro, así que un prefijo que calza no debe penalizarse. */
        private fun longestCommon(a: String, b: String): Int {
            if (a.isEmpty() || b.isEmpty()) return 0
            var best = 0
            val dp = IntArray(b.length + 1)
            for (i in a.indices) {
                var prev = 0
                for (j in b.indices) {
                    val tmp = dp[j + 1]
                    dp[j + 1] = if (a[i] == b[j]) prev + 1 else 0
                    best = max(best, dp[j + 1])
                    prev = tmp
                }
            }
            return best
        }
    }
}
