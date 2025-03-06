package furhatos.app.templateadvancedskill.flow.main

import furhatos.app.templateadvancedskill.flow.Parent
import furhatos.app.templateadvancedskill.flow.how_are_you.HowAreYou
import furhatos.app.templateadvancedskill.nlu.HowAreYouIntent
import furhatos.app.templateadvancedskill.nlu.NiceToMeetYouIntent
import furhatos.flow.kotlin.*
import furhatos.gestures.Gestures
import furhatos.nlu.Response
import furhatos.nlu.common.Greeting
import furhatos.util.Language

/**
 * Example state of a simple flow to greet a user.
 *
 */
// define the state as a function to be able to pass arguments to it
fun GreetUser(response: Response<*>? = null): State = state(Parent) {
    onEntry {
        if (response != null) raise(response) // raise any response that was passed on and handle the response here
        else furhat.listen() // or start a listen to collect a response in this state
    }
    // Handle partial responses where the user said a greeting and something else.
    onPartialResponse<Greeting> {
        furhat.say {
            random {
                +"Hi!"
                +"Hello!"
                +"Hi there!"
            }
        }
        // Raising the secondary intent will cause our triggers to handle the second part of the intent
        // Also raising the response (it) allows for acting on information in the response - e.g. what user spoke
        raise(it, it.secondaryIntent)
    }
    onResponse<Greeting> {
        furhat.say {
            random {
                +"Hi!"
                +"Hello!"
                +"Hi there!"
            }
        }
        goto(Active)
    }
    onResponse<HowAreYouIntent> {
        furhat.say {
            +"I feel"
            random {
                +"good"
                +"pretty good"
            }
            Gestures.BigSmile
        }
        call(HowAreYou) // We'll return the pleasantries, but then end the conversation.
        goto(Active)
    }
    onResponse<NiceToMeetYouIntent> {
        furhat.say {
            random {
                +"Nice too meet you too. "
                +"My pleasure. "
                +"Nice to see you as well. "
            }
            +Gestures.BigSmile
        }
        call(HowAreYou) // We'll return the pleasantries, but then end the conversation.
        goto(Active)
    }
    onResponse {
        goto(Active)
    }
    onNoResponse {
        goto(Active)
    }

}

/** Run this to test the intents of this state from the run terminal in IntelliJ. **/
fun main(args: Array<String>) {
    println("Type to test the intents of this state. (please ignore the initial error messages)")
    while (true) {
        println("Enter your user response...")
        val utterance = readLine()
        val results = GreetUser(null).getIntentClassifier(lang = Language.ENGLISH_US).classify(utterance!!)
        if (results.isEmpty()) {
            println("No match")
        } else {
            results.forEach {
                println("Matched ${it.intents} with ${it.conf} confidence")
            }
        }
    }
}