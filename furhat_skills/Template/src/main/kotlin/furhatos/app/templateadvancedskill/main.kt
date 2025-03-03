package furhatos.app.templateadvancedskill

import furhatos.app.templateadvancedskill.flow.Init
import furhatos.skills.Skill
import furhatos.flow.kotlin.*

class MyAdvancedSkill : Skill() {
    override fun start() {
        Flow().run(Init)
    }
}

fun main(args: Array<String>) {
    Skill.main(args)
}