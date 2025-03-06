package furhatos.app.templateadvancedskill.flow.how_are_you

import furhatos.flow.kotlin.*
import furhatos.util.Language


/**
 * Tip!
 *
 * This subflow has all relevant resources including intents, entities and phrases together in its own package
 * without any dependencies to other skills files. This allows for easy reuse of the subflow in other skills.
 */

/**
 * Example of a subflow
 * Flow will ask about how to user feels today and return.
 **/
val HowAreYou: State = state {
    onEntry {
        furhat.ask(phrases.howAreYou)
    }

    onResponse<PositiveReactionIntent> {
        val positiveWord: String? =
            it.intent.positiveExpressionEntity?.text // Check for what word the person used in the intent
        furhat.say(phrases.gladYouFeelGood(positiveWord))
        terminate()
    }

    onResponse {
        furhat.say(phrases.wellFeelingsAreComplex)
        delay(400)
        terminate()
    }

    onNoResponse {
        terminate()
    }
}

/** Run this to test the intents of this state from the command line. **/
fun main(args: Array<String>) {
    while (true) {
        val utterance = readLine()
        val results = HowAreYou.getIntentClassifier(lang = Language.ENGLISH_US).classify(utterance!!)
        if (results.isEmpty()) {
            println("No match")
        } else {
            results.forEach {
                println("Matched ${it.intents} with ${it.conf} confidence")
            }
        }
    }
}