package furhatos.app.templateadvancedskill.nlu

import furhatos.flow.kotlin.furhat
import furhatos.flow.kotlin.onResponse
import furhatos.flow.kotlin.partialState

/** Define universal commands and responses as partial states that can be included in a state. **/

val UniversalResponses = partialState {
    onResponse(listOf(HelpIntent(),WhatIsThisIntent())) {// Handle both HelpIntent and WhatIsThisIntent on the same trigger
        furhat.say {
                +"I'm sorry, "
            random {
                +"I don't know how to respond to that."
                +"I don't know what to say."
                +"I'm at a loss here. "
                +"I don't know what to say to that. "
            }
        }
    }
}