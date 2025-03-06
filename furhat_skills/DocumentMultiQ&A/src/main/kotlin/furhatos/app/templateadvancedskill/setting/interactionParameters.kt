package furhatos.app.templateadvancedskill.setting

import furhatos.records.Location
import furhatos.skills.SimpleEngagementPolicy
import furhatos.skills.UserManager

/** Engagement policies */
val defaultEngagementPolicy: SimpleEngagementPolicy
    get() = SimpleEngagementPolicy(
        userManager = UserManager,
        innerRadius = 1.0,
        outerRadius = 1.5,
        maxUsers = 2
    )

val sleepingEngagementPolicy: SimpleEngagementPolicy
    get() = SimpleEngagementPolicy(
        userManager = UserManager,
        innerRadius = 4.0,
        outerRadius = 4.0,
        maxUsers = 2
    )

val idleEngagementPolicy: SimpleEngagementPolicy
    get() = SimpleEngagementPolicy(
        userManager = UserManager,
        innerRadius = 2.0,
        outerRadius = 4.0,
        maxUsers = 10
    )

val activeEngagementPolicy: SimpleEngagementPolicy
    get() = SimpleEngagementPolicy(
        userManager = UserManager,
        innerRadius = 1.5,
        outerRadius = 2.5,
        maxUsers = 4
    )

/** Custom locations **/
val downMax = Location(0.0, -1.0, 3.0)

