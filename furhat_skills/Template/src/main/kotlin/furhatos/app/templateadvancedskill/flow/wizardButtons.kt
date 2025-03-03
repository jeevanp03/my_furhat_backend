package furhatos.app.templateadvancedskill.flow

import furhatos.app.templateadvancedskill.flow.how_are_you.HowAreYou
import furhatos.app.templateadvancedskill.flow.main.*
import furhatos.app.templateadvancedskill.setting.*
import furhatos.flow.kotlin.Color
import furhatos.flow.kotlin.Section
import furhatos.flow.kotlin.furhat
import furhatos.flow.kotlin.partialState

/** Universal button ta always be present **/
val UniversalWizardButtons = partialState {
    onButton("restart", color = Color.Red, section = Section.LEFT) {
        goto(Init)
    }
    onButton("Stop speaking", color = Color.Red, section = Section.LEFT) {
        furhat.stopSpeaking()
    }
}

/** Buttons to speed up testing **/
val TestButtons = partialState {
    onButton("Idle", color = Color.Blue, section = Section.RIGHT) {
        goto(Idle)
    }
    onButton("WaitingForEngagedUser", color = Color.Blue, section = Section.RIGHT) {
        goto(WaitingToStart)
    }
    onButton("Active", color = Color.Blue, section = Section.RIGHT) {
        goto(Active)
    }
    onButton("Greeting", color = Color.Blue, section = Section.RIGHT) {
        goto(GreetUser(null))
    }
    onButton("HowAreYou", color = Color.Blue, section = Section.RIGHT) {
        call(HowAreYou)
    }
    onButton("Nap", color = Color.Blue, section = Section.RIGHT) {
        goto(Nap)
    }
    onButton("DeepSleep", color = Color.Blue, section = Section.RIGHT) {
        goto(DeepSleep)
    }

    onButton("WakeUp", color = Color.Yellow, section = Section.RIGHT) {
        furhat.wakeUp()
    }
    onButton("WakeUp", color = Color.Yellow, section = Section.RIGHT) {
        furhat.fallASleep()
    }
    onButton("WakeUp", color = Color.Yellow, section = Section.RIGHT) {
        furhat.beIdle()
    }
    onButton("WakeUp", color = Color.Yellow, section = Section.RIGHT) {
        furhat.beActive()
    }

    onButton("set furhat persona", color = Color.Yellow, section = Section.RIGHT) {
        activate(furhatPersona)
    }
}