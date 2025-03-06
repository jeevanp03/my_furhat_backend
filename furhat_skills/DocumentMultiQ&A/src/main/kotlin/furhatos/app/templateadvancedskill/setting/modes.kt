package furhatos.app.templateadvancedskill.setting

import furhat.libraries.standard.GesturesLib
import furhatos.app.templateadvancedskill.flow.log
import furhatos.flow.kotlin.Furhat
import furhatos.flow.kotlin.dialogLogger
import furhatos.records.Location
import furhatos.skills.UserManager

/** Mode indicates the state of the robot.
 * It can be set in a dialogstate to trigger behaviour specific to the state of the robot.
 * E.g. LED lights, micro expressions, automatic head movements, etc
 **/
enum class Modes {
    IDLE,
    ACTIVE,
    SLEEPING
}

var mode = Modes.IDLE

/** Functions **/

/** Function to send the robot to sleep and waking it up. **/
fun Furhat.fallASleep() {
    if (mode != Modes.SLEEPING) { // Can only fall asleep from non sleeping mode
        log.info("Falling asleep. ")
        log.debug("Mode: Sleeping. ")
        mode = Modes.SLEEPING
        dialogLogger.endSession() // end dialog logging when there is no real conversation going on
        UserManager.engagementPolicy = sleepingEngagementPolicy
        attendNobody()
        gesture(GesturesLib.PerformFallAsleepPersist)
        attend(Location.DOWN)
    }
}

fun Furhat.wakeUp() {
    if (mode == Modes.SLEEPING) { // Can only wake up if sleeping.
        log.info("Waking up. ")
        log.debug("Mode: Active. ")
        mode = Modes.ACTIVE // We'll assume the robot will wake up to be active.
        dialogLogger.endSession() // end dialog logging when there is no real conversation going on
        gesture(GesturesLib.PerformWakeUpWithHeadShake)
    }
}

fun Furhat.beActive() {
    if (mode == Modes.ACTIVE) return // skip this step if robot is already active
    if (mode == Modes.SLEEPING) wakeUp() // wake up if sleeping
    log.debug("Mode: Active. ")
    mode = Modes.ACTIVE
    dialogLogger.startSession() // Start the dialogLogger when we have some actual conversation happening.
    UserManager.engagementPolicy = activeEngagementPolicy
    setMicroexpression(LISTENING_MICROEXPRESSIONS)
}

fun Furhat.beIdle() {
    if (mode == Modes.IDLE) return // skip this step if robot is already Idle
    if (mode == Modes.SLEEPING) wakeUp() // wake up if sleeping
    log.debug("Mode: Idling. ")
    mode = Modes.IDLE
    dialogLogger.endSession() // end dialog logging when there is no real conversation going on
    UserManager.engagementPolicy = idleEngagementPolicy
    setMicroexpression(DEFAULT_MICROEXPRESSIONS)
    attendNobody()
}

