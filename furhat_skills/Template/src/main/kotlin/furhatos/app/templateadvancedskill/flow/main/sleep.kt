package furhatos.app.templateadvancedskill.flow.main

import furhat.libraries.standard.NluLib
import furhatos.app.templateadvancedskill.flow.PowerSaving
import furhatos.app.templateadvancedskill.setting.fallASleep
import furhatos.app.templateadvancedskill.responses.gestures.reset
import furhatos.app.templateadvancedskill.setting.DeepSleepWhenNeglected
import furhatos.app.templateadvancedskill.setting.NapHeadMovements
import furhatos.app.templateadvancedskill.setting.downMax
import furhatos.app.templateadvancedskill.setting.isAttended
import furhatos.app.templateadvancedskill.setting.wakeUp
import furhatos.flow.kotlin.*
import furhatos.nlu.common.Greeting

/**
 * Tip!
 *
 * Use multiple sleep states to set more complex behavior when no users are present.
 *
 * */

/**
 * Light sleep. Listen for wakeup commands when users are present and looking at Furhat
 */
val Nap: State = state {
    include(NapHeadMovements)
    include(DeepSleepWhenNeglected)

    onEntry {
        furhat.fallASleep()
    }
    onReentry {
        if (furhat.isAttended()) {
            furhat.listen()
        }
    }
    onResponse<Greeting> {
        raise(NluLib.WakeUp()) // Raise another intent to handle pass it on to that trigger handler.
    }
    onResponse<NluLib.WakeUp> { // listen for either a wake-up- or a greeting intent.
        furhat.wakeUp()
        when {
            users.hasAny() -> goto(Active)
            !users.hasAny() -> goto(Idle)
        }
    }
    onUserAttend { // Whenever a user looks at the robot, it will start to listen
        reentry()
    }
    onResponse {
        reentry()
    }
    onNoResponse {
        reentry()
    }
    onNetworkFailed { // override (and silence) default fallback behaviour
        reentry()
    }
    onResponseFailed { // override (and silence) default fallback behaviour
        reentry()
    }
}

/**
 * Deep sleep. Can only wake up from wizard buttons in the web interface.
 */
val DeepSleep: State = state(PowerSaving) {
    onEntry {
        furhat.fallASleep()
        furhat.attend(downMax) // Make head fall even lower, when in "deep sleep".
        furhat.gesture(reset)
        delay(200)
        furhat.setVisibility(false, 3000) // Fade out face to black
    }
    onExit {
        furhat.setVisibility(true, 3000) // Fade in face
    }
}

