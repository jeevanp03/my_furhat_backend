package furhatos.app.templateadvancedskill.flow.how_are_you

import furhatos.flow.kotlin.Utterance
import furhatos.flow.kotlin.utterance
import furhatos.gestures.Gestures

val phrases = HowAreYouPhrases()

class HowAreYouPhrases {
    val feelGood: String = listOf("good", "pretty good").random()
    val feelGoodUtterance: Utterance
        get() = utterance {
            +"I feel"
            random {
                +"good"
                +"pretty good"
            }
            +Gestures.BigSmile
        }
    val howAreYou: Utterance
        get() = utterance {
            random {
                // use prosody tag for emphasis
                +"How are <prosody rate='85%'>you today?</prosody>"
                +"How are <prosody rate='85%'>you,</prosody> today?"
                +"How do <prosody rate='85%'>you,</prosody> feel today?"
            }
        }

    fun gladYouFeelGood(positiveWord: String? = "good"): Utterance {
        if (positiveWord != null) {
            return utterance {
                random {
                    +"Glad to hear you feel $positiveWord"
                    +"Nice that you feel $positiveWord"
                }
                delay(200)
            }
        } else {
            return utterance {
                random{
                    +"Glad to hear that. "
                    +"Nice to hear that. "
                }
            }
        }
    }

    val wellFeelingsAreComplex: Utterance
        get() = utterance {
            +Gestures.Thoughtful(0.6, 2.0)
            random {
                +"Well, feelings are complex."
                +"Well, feelings can be complex."
                +"I guess feelings can be complex."
            }
        }
}

