package furhatos.app.templateadvancedskill.flow.main

import furhatos.flow.kotlin.*
import furhatos.app.templateadvancedskill.flow.Parent

val DocumentWaitingToStart: State = state(parent = Parent) {
    onEntry {
        furhat.ask("I'm ready to assist with your document questions. Could you please tell me what subject you're interested in, or simply the name of the document?")
    }

    // When any response is detected, transition to SelectDocument.
    onResponse {
//        goto(SelectDocument)
        val userInput = it.text.trim()
        // Call the API endpoint /get_docs to perform document retrieval/classification.
        val bestDocName = callGetDocs(userInput)
        // Transition to the Q&A state for the matched document.
        goto(documentInfoQnA(bestDocName))
    }

    onNoResponse {
        furhat.ask("I didn't catch that. Please tell me the subject or the name of the document you're interested in.")
        reentry()
    }
}
