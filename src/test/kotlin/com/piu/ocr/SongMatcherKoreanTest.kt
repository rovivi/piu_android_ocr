package com.piu.ocr

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

/**
 * Lecturas Hangul contra el título "k" del catálogo. Hoy chars.bin no tiene
 * plantillas Hangul, así que esto prueba el camino listo para cuando las tenga,
 * y que el latino no cambió.
 */
class SongMatcherKoreanTest {

    private val matcher = SongMatcher(File("src/main/assets/piu_ocr/catalog.json").readText())

    @Test
    fun aHangulReadMatchesTheKoreanTitleAndAnswersWithTheDisplayName() {
        assertEquals("Freedom Dive", matcher.match("프리덤 다이브").first().name)
        // Una sílaba mal leída sigue siendo esta canción.
        assertEquals("Freedom Dive", matcher.match("프리덤 다이부").first().name)
    }

    @Test
    fun hangulSurvivesNormalizeAndLatinIsUnchanged() {
        assertEquals("프리덤 다이브 short cut", SongMatcher.normalize("프리덤 다이브 - SHORT CUT -"))
        assertEquals("fracture temporelle e", SongMatcher.normalize("Fracture Temporelle É!"))
    }
}
