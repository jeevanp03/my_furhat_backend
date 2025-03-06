package furhatos.app.templateadvancedskill.flow.main

import furhat.libraries.standard.BehaviorLib.AutomaticMovements.randomHeadMovements
import furhatos.app.templateadvancedskill.flow.Global
import furhatos.app.templateadvancedskill.flow.log
import furhatos.app.templateadvancedskill.setting.*
import furhatos.flow.kotlin.*

/**
 * Tip!
 *
 * Use simple Idle state where the interaction will start as soon as a user enters for situations where you know
 * people want to be interacting with the robot. Or define a more complex idle state like "WaitingForEngagedUser"
 * to avoid starting the interaction without the user actually being interested to engage with the user.
 */

/** State where Furhat is inactive with no users in front of it */
val Idle: State = state(parent = Global) {
    include(randomHeadMovements())
    include(napWhenTired) // Sends the robot to "sleep" after a period of inactivity

    init { // init is only performed once
        if (furhat.isVirtual()) furhat.say("Add a virtual user to start the interaction")
    }

    onEntry { // on entry performed every time entering the state
        furhat.beIdle() // custom function to set the robot in a specific 'mode'
        log.info("idling")
    }

    onUserEnter {
        furhat.attend(it) // It's good practice to make attention shifts on the trigger, instead of shifting the attention onEntry in the state.
        goto(WaitingToStart)
    }
}

/**
 * State where Furhat is waiting for users to walk up close to the robot and engage with it.
 * While waiting Furhat will seek their attention and switch attention between users.
 * */
val WaitingToStart: State = state(parent = Global) {
    include(randomHeadMovements())
    include(SeekAttention) // Attention seeking behaviour
    include(napWhenTired) // Go to sleep after a while if no users are engaging

    onEntry {
        activate(furhatPersona)
        when {
            users.hasAny() -> {
                log.debug("User present - starting document interaction.")
                furhat.attend(users.random)
                goto(furhatos.app.templateadvancedskill.flow.main.DocumentWaitingToStart)
            }
            else -> {
                log.debug("No users present - idling.")
                goto(Idle)
            }
        }
    }

    onReentry {
        when {
            !users.hasAny() -> goto(Idle)
            furhat.isAttended() -> goto(Active)
            else -> log.debug("keep idling") // do nothing particular - just keep idling
        }
    }
    onUserAttend {
        log.debug("user ${it.id} attended in ${thisState.name}")
        furhat.attend(it)
        goto(Active)
    }
    onUserLeave(instant = true) {
        when { // "it" is the user that left
            !users.hasAny() -> goto(Idle) // no more users
            furhat.isAttending(it) -> furhat.attend(users.nextMostEngagedUser()) // current user left
            else -> furhat.glance(it.head.location) // other user left, just glance
        }
    }

}

