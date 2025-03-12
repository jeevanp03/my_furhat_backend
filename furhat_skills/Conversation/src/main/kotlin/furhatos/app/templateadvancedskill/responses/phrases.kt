package furhatos.app.templateadvancedskill.responses

import furhatos.flow.kotlin.Utterance
import furhatos.flow.kotlin.utterance
import furhatos.gestures.Gestures

/**
 * Tip!
 * Define phrases elsewhere (here) to de-clutter the flow and keep all the dialogue in one place.
 * Useful for when creating multi-language skills, or skills with multiple personas with the same script.
 *
 * See furhatos.app.templateadvancedskill.responses.phrases for an example.
 * furhat.say(phrases.feelGoodUtterance)
 */

class Phrases {
    val Q_GoOnPowerBreak = listOf(
        "Did you want me to go on a power break?",
        "Do you want me to take a power nap? ",
        "Is it time for a power nap? ",
        "Should I take a short Power break?"
    ).random() // Include variance in phrasing without cluttering the flow.
    val Q_NapTimeOver = listOf(
        "Is nap time over?",
        "Is the nap over?",
        "Is the time for napping over?",
        "Is it time to stop napping?"
    ).random() // Include variance in phrasing without cluttering the flow.
    val A_feelGoodUtterance: Utterance = utterance { // Define complete and more complex utterances to get more variance and rich expressions without cluttering the flow.
        +"I feel"
        random {
            +"good"
            +"pretty good"
        }
        +Gestures.BigSmile
    }
}

val phrases = Phrases()