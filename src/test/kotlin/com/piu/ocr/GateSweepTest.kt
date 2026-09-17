package com.piu.ocr

import org.json.JSONObject
import org.junit.Test
import java.io.File

/**
 * Not a test: a dump. For every fixture photo, the ground-truth title against the matcher's pooled
 * candidates, so a gate (margin, absolute floor) can be chosen from data instead of by feel.
 * Writes build/gate_sweep.tsv.
 */
class GateSweepTest {
    @Test
    fun dump() {
        val fixture = JSONObject(File("src/test/resources/parity_fixture.json").readText())
        val matcher = SongMatcher(File("src/main/assets/piu_ocr/catalog.json").readText())
        val out = File("build/gate_sweep.tsv")
        out.parentFile.mkdirs()
        out.printWriter().use { w ->
            w.println("key\tgt\tsong\ttop1\ts1\ttop2\ts2\tmargin\tglyphs\traw")
            val rows = fixture.getJSONArray("rows")
            for (i in 0 until rows.length()) {
                val r = rows.getJSONObject(i)
                val gt = r.getJSONObject("gt").optString("song", "")
                val reading = PiuOcr.interpret(r.getJSONObject("native").toString(), matcher)
                val c = reading.candidates
                val s1 = c.getOrNull(0)?.score ?: 0.0
                val s2 = c.getOrNull(1)?.score ?: 0.0
                val raw = reading.rawTitle.replace('\t', ' ')
                w.println(
                    listOf(
                        r.getString("key"), gt, reading.song.value ?: "",
                        c.getOrNull(0)?.name ?: "", "%.4f".format(s1),
                        c.getOrNull(1)?.name ?: "", "%.4f".format(s2),
                        "%.4f".format(s1 - s2),
                        raw.replace(" | ", "").length.toString(),
                        raw,
                    ).joinToString("\t"),
                )
            }
        }
    }
}
