package furhatos.app.templateadvancedskill.responses.gestures

import furhatos.gestures.BasicParams
import furhatos.gestures.CharParams
import furhatos.gestures.Gestures
import furhatos.gestures.defineGesture

/** Define your custom gestures. Note there are more gestures in the Asset Collection furhat.libraries.standard.GesturesLib **/

val hearSpeechGesture = defineGesture("hearSpeechGesture") {
    frame(0.4, persist = true) {
        CharParams.EYEBROW_UP to 1.0
    }
    reset(2.5)
}

// Raise the eyebrows slightly instead to signal interest
val Listening = defineGesture("Listening") {
    frame(0.4, persist = true) {
        Gestures.BrowRaise to 0.8
        Gestures.Smile to 0.6
    }
}

val Normal = defineGesture("Normal") {
    frame(0.4, persist = true) {
        Gestures.BrowRaise to 0.0
        Gestures.Smile to 0.0
    }
    reset(0.5)
}

val reset = defineGesture("reset") {
    frame(0.3) {
        BasicParams.NECK_PAN to 0.0
        BasicParams.NECK_ROLL to 0.0
        BasicParams.NECK_TILT to 0.0
    }
    reset(0.4)
}