package furhatos.app.templateadvancedskill.flow

import furhatos.libraries.standard.BehaviorLib.AutomaticMovements.randomHeadMovements
import furhatos.app.templateadvancedskill.flow.main.Idle
import furhatos.app.templateadvancedskill.flow.main.WaitingToStart
import furhatos.app.templateadvancedskill.nlu.UniversalResponses
import furhatos.app.templateadvancedskill.setting.UniversalFallbackBehaviour
import furhatos.app.templateadvancedskill.setting.nextMostEngagedUser
import furhatos.app.templateadvancedskill.setting.wakeUp
import furhatos.flow.kotlin.*
import java.util.concurrent.TimeUnit
import furhatos.app.templateadvancedskill.nlu.AskForDocumentInfo

// Global flag to lock the attended user during specific interactions.
var lockAttention: Boolean = false

/**
 * Parent state that houses global definitions that should always be active in any state in the main flow.
 */
val Global: State = state {
    include(UniversalWizardButtons) // Wizard buttons that should be available in all states
    include(UniversalResponses)     // Commands that users should always be able to say.
    include(UniversalFallbackBehaviour) // Set default fallback behaviour
    if (testMode) include(TestButtons)   // Wizard buttons for easy testing
}

/**
 * Additional more specific parent state.
 * Use for all states where the robot is being actively engaged in the interaction.
 */
val Parent: State = state(parent = Global) {
    // Include random head movements for natural behavior.
    include(randomHeadMovements())

    onUserEnter(instant = true) {
        if (!lockAttention) {
            when {
                furhat.isAttendingUser -> furhat.glance(it) // Glance at new users entering.
                else -> furhat.attend(it) // Attend user if not attending anyone.
            }
        } else {
            // When attention is locked, simply glance.
            furhat.glance(it)
        }
        reentry()
    }

    onUserLeave(instant = true) {
        if (!lockAttention) {
            when {
                !users.hasAny() -> goto(Idle) // No more users.
                furhat.isAttending(it) -> furhat.attend(users.nextMostEngagedUser()) // If current user leaves, attend the next.
                else -> furhat.glance(it.head.location) // Other user leaves, just glance.
            }
        } else {
            // When attention is locked, do not reassign the attended user.
            furhat.glance(it.head.location)
        }
        reentry()
    }

    // Add this block to globally catch AskForDocumentInfo and transition to DocumentWaitingToStart.
    onResponse {
        goto(furhatos.app.templateadvancedskill.flow.main.DocumentWaitingToStart)
    }

    onExit(inherit = true) {
        // Reset local counters.
        noMatch = 0
        noInput = 0
    }
}

/** Sleeping state: use for states where the robot is in power-save mode and can only be woken up by wizard buttons. */
val PowerSaving: State = state(parent = Global) {
    onEntry(inherit = true) {
        delay(5, TimeUnit.SECONDS) // Stay asleep for at least 5 seconds.
    }
    onButton("Wake Up") {
        furhat.wakeUp()
        if (users.hasAny()) goto(WaitingToStart) else goto(Idle)
    }
}
