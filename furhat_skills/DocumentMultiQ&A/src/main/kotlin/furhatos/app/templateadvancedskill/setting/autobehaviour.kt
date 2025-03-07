package furhatos.app.templateadvancedskill.setting

import furhat.libraries.standard.BehaviorLib.AutomaticMovements.straightHead
import furhatos.app.templateadvancedskill.flow.log
import furhatos.app.templateadvancedskill.flow.main.DeepSleep
import furhatos.app.templateadvancedskill.flow.main.Idle
import furhatos.app.templateadvancedskill.flow.main.Nap
import furhatos.app.templateadvancedskill.flow.main.WaitingToStart
import furhatos.app.templateadvancedskill.flow.noInput
import furhatos.app.templateadvancedskill.flow.noMatch
import furhatos.app.templateadvancedskill.utils.NoUsersPresentEvent
import furhatos.app.templateadvancedskill.utils.RandomHeadEvent
import furhatos.autobehavior.defineMicroexpression
import furhatos.flow.kotlin.*
import furhatos.gestures.BasicParams
import furhatos.gestures.Gestures
import furhatos.gestures.defineGesture
import furhatos.records.Location
import java.util.concurrent.TimeUnit

/**
 * Parameters to control robot behaviour
 */

/** Timeouts for napping and sleeping*/
// Timeout for inactivity before Furhat takes a nap
val WAIT_TIME_TO_NAP = TimeUnit.SECONDS.toMillis(300).toInt()

// Timeout for inactivity in napping before going to a deep sleep / "power save" mode.
val NAP_TIME_BEFORE_DEEP_SLEEP = TimeUnit.SECONDS.toMillis(600).toInt()

/** Attention switching **/
// Min and max time to attend a user. Use for knowing when to switch attention.
const val MIN_TIME_TO_ATTEND_USER = 3000 // milliseconds
val MAX_TIME_TO_ATTEND_USER = TimeUnit.SECONDS.toMillis(15).toInt()

// Timeout for how long to keep attending nobody when there are users to attend.
val ATTEND_NOBODY_TIMEOUT = TimeUnit.SECONDS.toMillis(100).toInt()
val ATTEND_OTHER_TIMEOUT = TimeUnit.SECONDS.toMillis(30).toInt()

// Timeout for how often to look away
const val STARE_TIMEOUT = 15000 //milliseconds

/** Attention parameters **/
//  Set angle when users are no longer considered attending Furhat. These are the default parameters set in the Skill Framework.
const val DEFAULT_ATTENTION_GAINED_THRESHOLD = 20.0 // angle
const val DEFAULT_ATTENTION_LOST_THRESHOLD = 35.0 // angle

// How long a user has to attend Furhat before the user is considered 'engaged' */
const val ENGAGED_ATTEND_TIME = 2300 //milliseconds

/**
 * Behaviors to include in states
 */

/** avoid too long awkward silences **/
val AutoUserAttentionSwitching = partialState {
    val offset = 1000
    onTime(
        delay = MAX_TIME_TO_ATTEND_USER,
        repeat = (MAX_TIME_TO_ATTEND_USER - offset)..(MAX_TIME_TO_ATTEND_USER + offset), // Interval of how often to repeat
        instant = true
    ) {
        if (users.hasCurrent()) { // There is a user that furhat is attending
            when {
                users.current.isAttendingFurhat -> { // user is looking at furhat, and furhat is looking at user
                    //do nothing
                    log.debug("keeping attention on user")
                }
                users.usersAttendingFurhat.isNotEmpty() -> { // no one is attending furhat we are expected to return to wait for user attention
                    goto(WaitingToStart)
                }
                !furhat.isAttendingUser -> {
                    furhat.attendOtherOrNobody()
                    furhat.attend(users.nextMostEngagedUser())
                }
            }
        }
    }
}


/** Avoid staring **/
val AutoGlanceAway = partialState {
    val offset = 500
    onTime(
        delay = STARE_TIMEOUT,
        repeat = (STARE_TIMEOUT - offset)..(STARE_TIMEOUT + offset), // Interval of how often to repeat
        instant = true
    ) {
        log.info("breaking eye contact")
        furhat.glance(Location.DOWN)
    }
}

/**
 * Seek attention among user present in the interaction space but not engaged with Furhat.
 */
val SeekAttention = partialState {
    onTime(
        delay = ATTEND_OTHER_TIMEOUT,
        repeat = ATTEND_OTHER_TIMEOUT,
        instant = true
    ) {
        log.info("On time: Re-evaluate what user to attend")
        when {
            // 1. No users to seek attention from? Raise an event for the flow to handle.
            users.count == 0 -> raise(NoUsersPresentEvent())  //
            // 2. Furhat is not attending anyone - find someone to attend
            !furhat.isAttendingUser -> {
                log.debug("Furhat is idling, not attending a user")
                furhat.attend(users.nextMostEngagedUser())
            }

            // 3. Furhat is already attending a user - consider switch attention to new user, or to take a break from looking at people
            furhat.isAttendingUser -> {
                log.debug("Furhat is attending a user.")
                when (users.count) {
                    1 -> furhat.attendNobody()
                    else -> furhat.attend(users.other) // find another user to attend
                }
            }
        }
        reentry()
    }
    onTime(
        delay = ATTEND_NOBODY_TIMEOUT,
        repeat = ATTEND_NOBODY_TIMEOUT,
        instant = true
    ) {
        log.info("Take a break from looking at people. ")
        furhat.attendNobody()
    }
}

// Timer to trigger the robot to go to sleep.
val napWhenTired = partialState {
    onTime(WAIT_TIME_TO_NAP) {
        goto(Nap)
    }
}

// Timer to trigger the robot to go to deep sleep.
val DeepSleepWhenNeglected = partialState {
    onTime(NAP_TIME_BEFORE_DEEP_SLEEP) {
        goto(DeepSleep)
    }
}

/**
 * Universal fallback behavior for managing issues in the flow gracefully.
 * The current configuration is a replica of the default behaviour. Remove this, and the robot will default to the same exact behavior.
 */
val UniversalFallbackBehaviour = partialState {

    onNoResponse {
        furhat.say("Sorry I didn't hear you. ")
        reentry()
    }
    onNoResponse() {
        noInput++
        when (noInput) {
            0 -> furhat.say("Sorry I didn't hear you. ")
            else -> furhat.say("Sorry I still didn't hear you. ")
        }
        reentry()
    }
    onResponse {
        noMatch++
        when (noMatch) {
            0 -> furhat.say("Sorry, I didn't understand that. ")
            else -> furhat.say("Sorry, I still didn't understand that. ")
        }
        reentry()
    }
    // Override default error messages on internet failure
    onResponseFailed {
        furhat.say("Sorry, I am having trouble with the internet connection.")
        reentry()
    }
    onNetworkFailed {
        furhat.say("Sorry, I am experiencing trouble with the internet connection.")
        reentry()
    }
    onEvent<NoUsersPresentEvent> {
        goto(Idle)
    }
    onExit(inherit = true) {
        noMatch = 0
        noInput = 0
    }
}

/** Custom nap head movements */
val NapHeadMovements = partialState {
    onTime(2000..2500, 5200..9000, instant = true) {
        raise(RandomHeadEvent())
    }
    onExit {
        furhat.gesture(straightHead)
    }
    onEvent<RandomHeadEvent>(instant = true) {
        if (enableNapHeadMovements) {
            // Regulates when Furhat does idle head movements
            val gesture = napHeadMovements(
                // these overrides the ones defined in idleheadmovements, was also affecting other functions that have been removed. This is kept if we want to expand this in the future.
                strength = 1.0,
                duration = 2.0, // Affect how fast the idle head movements will be
                amplitude = 5.0, // How big the head movements will be
                gazeAway = false // Function removed, but kept here if we re-introduce
            )
            furhat.gesture(gesture)
        }
    }
}
var enableNapHeadMovements = true

fun getRandomDirection(): Double {
    // Randomizes if Furhat is looking left/right and up/down
    return if (Math.random() < 0.5)
        -1.0 // negative
    else
        +1.0 // positive
}

fun getScaleParameter(): Double {
    return getRandomDirection() * (Math.random() + 0.55)
    // Adds variation to the amplitude of the random head movements and sets the range
}

fun napHeadMovements(
    strength: Double = 1.0,
    duration: Double = 1.0,
    amplitude: Double = 5.0,
    gazeAway: Boolean = false
) =
    defineGesture("headMove", strength = strength, duration = duration) {
        frame(1.5, 8.5) {
            // This regulates how long the position will be held and how fast Furhat gets there. Multiplied with duration.
            BasicParams.NECK_TILT to amplitude * getScaleParameter()
            BasicParams.NECK_ROLL to amplitude * getScaleParameter()
            BasicParams.NECK_PAN to 0.0
            // the direction (tilt/roll/pan) of the position shift is completely random. Does not affect attention
        }
        reset(10.0)
    }

val DEFAULT_MICROEXPRESSIONS = defineMicroexpression {
    // Fluctuate facial movements. First parameter is frequency, second amplitude, third adjustment.
    fluctuate(0.025, 0.06, 0.12, BasicParams.BROW_UP_LEFT, BasicParams.BROW_UP_RIGHT)
    fluctuate(0.025, 0.2, 0.5, BasicParams.SMILE_CLOSED)
    fluctuate(0.025, 0.2, 0.5, BasicParams.EXPR_SAD)
    // Note! We have to set all values defined by other MicroExpressions to make sure we are overwriting them
    fluctuate(0.025, 0.0, 0.0, BasicParams.EYE_SQUINT_LEFT, BasicParams.EYE_SQUINT_RIGHT)
    fluctuate(0.025, 0.0, 0.0, BasicParams.PHONE_BIGAAH)
    // Adjust eye gaze randomly (between -3 and 3 degrees) with a random interval of 200-400 ms.
    repeat(200..400) {
        adjust(-3.0..3.0, BasicParams.GAZE_PAN)
        adjust(-3.0..3.0, BasicParams.GAZE_TILT)
    }
    // Blinking with a random interval of 2-8 seconds
    repeat(2000..8000, Gestures.Blink)
}

val LISTENING_MICROEXPRESSIONS = defineMicroexpression {
    // Fluctuate facial movements. First parameter is frequency, second amplitude, third adjustment.
    fluctuate(0.025, 0.4, 0.2, BasicParams.BROW_UP_LEFT, BasicParams.BROW_UP_RIGHT)
    fluctuate(0.025, 0.4, 0.6, BasicParams.SMILE_CLOSED)
    // Note! We have to set all values defined by other MicroExpressions to make sure we are overwriting them
    fluctuate(0.025, 0.0, 0.0, BasicParams.EXPR_SAD)
    fluctuate(
        0.025,
        -0.2,
        0.2,
        BasicParams.EYE_SQUINT_LEFT,
        BasicParams.EYE_SQUINT_RIGHT
    )
    fluctuate(0.025, 0.05, 0.08, BasicParams.PHONE_BIGAAH)
    // Adjust eye gaze randomly (between -3 and 3 degrees) with a random interval of 200-400 ms.
    repeat(200..400) {
        adjust(-3.0..3.0, BasicParams.GAZE_PAN)
        adjust(-3.0..3.0, BasicParams.GAZE_TILT)
    }
    // Blinking with a random interval of 2-6 seconds
    repeat(2000..6000, Gestures.Blink)
}