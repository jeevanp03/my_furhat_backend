package furhatos.app.templateadvancedskill.setting

import furhatos.app.templateadvancedskill.flow.log
import furhatos.flow.kotlin.Furhat
import furhatos.records.Location
import furhatos.records.User
import furhatos.skills.UserManager

/**
 * Extensions and support functions for managing users. Many functions can/will be incorporated into the Asset Collection,
 * or as native function directly in the Kotlin skill API.
 **/

/** Extensions for user manager */
val UserManager.usersAttendingFurhat: List<User>
    get() = this.list.filter { it.isAttendingFurhat }
        .sortedBy { it.head.location.distance(Location.ORIGIN) } // all users looking at furhat sorted by the closest user
val UserManager.userClosestToFurhat: User get() = this.userClosestToPosition(Location.ORIGIN)
fun UserManager.nextMostEngagedUser(): User {
    // Signs of engagement
    // 1. Looks at the robot
    // 2. User standing closest
    when (count) {
        1 -> return list.first() // return the first (only) user.
        else -> when (usersAttendingFurhat.count()) {
            0 -> return userClosestToFurhat
            else -> return usersAttendingFurhat.first() // the list is sorted by the closest user first
        }
    }
}

fun Furhat.isAttended(): Boolean {
    for (user in users.list) {
        if (user.isAttendingFurhat()) {
            return true
        }
    }
    return false
}

/**
 * Convenience function when switching attention.
 * Attend another user if available, if not: stop attending.
 */
fun Furhat.attendOtherOrNobody() {
    when {
        users.count < 2 -> {
            attendNobody()
            log.info("attending nobody")
        }
        else -> {
            attend(users.other)
            log.info("attending other")
        }
    }
}
