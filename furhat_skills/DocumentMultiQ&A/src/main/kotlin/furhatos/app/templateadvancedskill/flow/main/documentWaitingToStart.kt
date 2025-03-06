package furhatos.app.templateadvancedskill.flow.main

import furhatos.flow.kotlin.*
import furhatos.app.templateadvancedskill.flow.Parent

val DocumentWaitingToStart: State = state(parent = Parent) {
    onEntry {
        furhat.say("I'm ready to assist with your document questions. Could you please tell me what subject you're interested in, or simply the name of the document?")
    }

    // When any response is detected, transition to SelectDocument.
    onResponse {
        goto(SelectDocument)
    }

    onNoResponse {
        furhat.say("I didn't catch that. Please tell me the subject or the name of the document you're interested in.")
        reentry()
    }
}
