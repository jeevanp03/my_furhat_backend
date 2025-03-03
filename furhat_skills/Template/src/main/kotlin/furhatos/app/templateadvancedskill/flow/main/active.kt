package furhatos.app.templateadvancedskill.flow.main

import furhatos.app.templateadvancedskill.flow.Parent
import furhatos.app.templateadvancedskill.flow.log
import furhatos.app.templateadvancedskill.nlu.HowAreYouIntent
import furhatos.app.templateadvancedskill.nlu.NiceToMeetYouIntent
import furhatos.app.templateadvancedskill.responses.gestures.hearSpeechGesture
import furhatos.app.templateadvancedskill.setting.AutoGlanceAway
import furhatos.app.templateadvancedskill.setting.AutoUserAttentionSwitching
import furhatos.app.templateadvancedskill.setting.beActive
import furhatos.app.templateadvancedskill.setting.isAttended
import furhatos.flow.kotlin.*
import furhatos.nlu.common.Greeting

/**
 * State where Furhat engage actively with the user.
 * Start your interaction from here.
 */
val Active: State = state(Parent) {
    onEntry {
        furhat.beActive()
        log.debug("now I'm listening")

        // We're leaving the initiative to the user and extending the listen timeout from default 5000 ms to 8000 ms.
        furhat.listen(8000)
    }
    onReentry {
        when {
            !users.hasAny() -> goto(Idle)
            !furhat.isAttended() -> goto(WaitingToStart)
            else -> furhat.listen()
        }
    }
    include(AutoUserAttentionSwitching) // Switch user after a while
    include(AutoGlanceAway) // Glance away after some time of eye contact

    /** Handle simple meet and greet conversation **/
    // Handle multiple intents where one part of the intent was a Greeting
    onPartialResponse<Greeting> {
        furhat.attend(it.userId) // attend the user that spoke
        goto(GreetUser(it))
    }
    onResponse(listOf(Greeting(), HowAreYouIntent(), NiceToMeetYouIntent())) {
        furhat.attend(it.userId) // attend the user that spoke
        goto(GreetUser(it))
    }
    /** Handle other (or no) user responses **/
    onNoResponse {
        // On no response we let the initiative remain with the user and just keep listening passively. 
        // This overrides the default behavior: "Sorry I didn't hear you "
        reentry()
    }
    onResponse {
        // On unknown response, the robot reacts to the user speaking, but doesn't engage and take the initiative.
        // This overrides the default behavior: "Sorry I didn't understand that"
        furhat.attend(it.userId) // attend the user that spoke
        furhat.gesture(hearSpeechGesture)
        reentry()
    }
    /** Handle Attention switching, see also default attention behaviour in parent state **/
    onUserAttend {
        furhat.attend(it)
        reentry()
    }

}