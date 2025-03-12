package furhatos.app.templateadvancedskill.flow.main

import java.time.LocalDateTime
import java.time.ZoneId
import java.util.*

val localDateTime: LocalDateTime = Date().toInstant().atZone(ZoneId.systemDefault()).toLocalDateTime()
val dayOfWeek = localDateTime.dayOfWeek.name
val hourOfDay = localDateTime.hour
val timeOfDay = when (hourOfDay) {
    in 0..3 -> "night"
    in 4..12 -> "morning"
    in 13..18 -> "afternoon"
    in 19..21 -> "evening"
    in 22..23 -> "night"
    else -> ""
}
// furhat.say("Nice to meet you this lovely ${dayOfWeek} ${timeOfDay}")