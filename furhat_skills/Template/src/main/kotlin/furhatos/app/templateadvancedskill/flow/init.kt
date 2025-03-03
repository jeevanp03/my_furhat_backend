package furhatos.app.templateadvancedskill.flow

import furhatos.app.templateadvancedskill.MyAdvancedSkill
import furhatos.app.templateadvancedskill.flow.main.Idle
import furhatos.app.templateadvancedskill.flow.main.WaitingToStart
import furhatos.app.templateadvancedskill.flow.paralell.InteractionGlow
import furhatos.app.templateadvancedskill.setting.*
import furhatos.app.templateadvancedskill.utils.loadProperties
import furhatos.autobehavior.bigSmileProbability
import furhatos.autobehavior.enableSmileBack
import furhatos.autobehavior.smileBlockDelay
import furhatos.autobehavior.smileProbability
import furhatos.flow.kotlin.State
import furhatos.flow.kotlin.furhat
import furhatos.flow.kotlin.state
import furhatos.flow.kotlin.users
import furhatos.nlu.LogisticMultiIntentClassifier
import furhatos.util.CommonUtils
import java.io.File

/** Flow parameters */
var testMode: Boolean = false
var noMatch = 0
var noInput = 0

// Get the log4J logger available from furhat.system.logger to print messages to the IntelliJ console and the log console in the web interface.
// Set the logging level in skill.properties
val log = CommonUtils.getLogger(MyAdvancedSkill::class.java)

/** State to initiate the skill, setting interaction parameters and starting the dialogue flow **/
val Init: State = state {
    init {
        /** Load properties file **/
        loadProperties()

        /** Set testMode for speedy testing */
        testMode = true

        /** define listening parameters */
        furhat.param.endSilTimeout = 800 //milliseconds
        furhat.param.noSpeechTimeout = 5000 //milliseconds
        furhat.param.maxSpeechTimeout = 15000 //milliseconds

        /** Set our default interaction parameters */
        users.engagementPolicy = defaultEngagementPolicy

        /** define our default engagement parameters */
        users.attentionGainedThreshold = DEFAULT_ATTENTION_GAINED_THRESHOLD
        users.attentionLostThreshold = DEFAULT_ATTENTION_LOST_THRESHOLD

        /** define our default smile back behavior */
        furhat.enableSmileBack = true
        smileProbability = 0.4
        bigSmileProbability = 0.3
        smileBlockDelay = 5000//milliseconds

        /** enable alternate intent classifier
        see: https://docs.furhat.io/nlu/#alternative-classification */
        LogisticMultiIntentClassifier.setAsDefault()

        /** Set default microepressions **/
        furhat.setMicroexpression(DEFAULT_MICROEXPRESSIONS)

        /** Initiate loggers **/
        // Dialog logger is enabled when there is dialog happening - see "beActive()"
        // The flow logger will dump a detailed log of the flow.
        // Use the general log4J logger to print log messages for development purposes.
        if (testMode) {
            //Start a flow logger to dump all flow actions into a logfile.
            val flowLoggerFile = File("logs/flowlog.txt") //Log file can have any extension.
            flowLogger.start(flowLoggerFile)
        }

        /** Start parallel flow to manage the LED **/
        parallel(abortOnExit = false) { goto(InteractionGlow) }
    }
    onEntry {
        /** Set our main character - defined in personas */
        activate(furhatPersona)

        /** start interaction */
        when {
            users.hasAny() -> {
                log.debug("User present - start the interaction. ")
                furhat.attend(users.random)
                goto(WaitingToStart)
            }
            else -> {
                log.debug("No users present - idling. ")
                goto(Idle) // Consider starting the interaction in Sleep.
            }
        }
    }
}

